from __future__ import annotations

import json
import uuid
from typing import Any

from .config import Settings
from .db import Database
from .document_editor import SmartDocumentEditor
from .memory import Memory
from .planner import OllamaPlanner, RulePlanner
from .tools import ToolRegistry


class AgentRuntime:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.db = Database(settings.database_path)
        self.editor = SmartDocumentEditor(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
        )
        self.tools = ToolRegistry(self.db, editor=self.editor)
        self.memory = Memory(self.db)
        self.fallback_planner = RulePlanner()
        if settings.planner == "ollama":
            self.planner = OllamaPlanner(
                settings.ollama_base_url,
                settings.ollama_model,
                settings.ollama_timeout,
            )
        else:
            self.planner = self.fallback_planner

    def run(self, session_id: str, user_id: str, text: str, doc_id: str | None = None) -> dict[str, Any]:
        self.db.add_message(session_id, user_id, "user", text)
        self.db.add_event(session_id, "user_message", {"text": text, "user_id": user_id, "doc_id": doc_id})

        context = self.memory.context(session_id, user_id)
        if doc_id:
            active_doc = self.db.get_document(doc_id)
            if active_doc:
                context["active_document"] = active_doc

        trace: list[dict[str, Any]] = []

        skills = self.tools.skill_instructions()
        try:
            plan = self.planner.plan(text, context, self.tools.descriptions(), skills=skills)
        except Exception as exc:
            # Fallback to rule-based planner if remote model is unreachable/returns error
            plan = self.fallback_planner.plan(text, context, self.tools.descriptions(), skills=skills)
            trace.append({"type": "warning", "summary": f"Remote planner error ({exc}), fell back to rule planner."})
        if plan.summary:
            trace.append({"type": "thought", "summary": plan.summary})
        self.db.add_event(session_id, "plan", {"intent": plan.intent, "summary": plan.summary})

        if plan.tool_name:
            tool = self.tools.get(plan.tool_name)
            args = plan.arguments or {}
            # If doc_id was passed in context and not in arguments, inject it for document tools
            if doc_id and "doc_id" in tool.parameters and not args.get("doc_id"):
                args["doc_id"] = doc_id

            if tool.requires_confirmation:
                token = f"appr-{uuid.uuid4().hex[:12]}"
                self.db.create_approval(token, session_id, user_id, plan.tool_name, args)
                reply = (
                    plan.reply
                    or f"L'action '{plan.tool_name}' nécessite une confirmation pour être exécutée."
                )
                self.db.add_message(session_id, user_id, "assistant", reply)
                self.db.add_event(
                    session_id,
                    "approval_requested",
                    {"token": token, "tool_name": plan.tool_name, "arguments": args},
                )
                active_doc_id = args.get("doc_id") or doc_id
                target_doc = self.db.get_document(active_doc_id) if active_doc_id else None
                return {
                    "session_id": session_id,
                    "reply": reply,
                    "status": "approval_required",
                    "approval_token": token,
                    "trace": trace,
                    "doc_id": active_doc_id,
                    "document": target_doc,
                }

            # Execute safe read action immediately
            result = tool.handler(**args, user_id=user_id)
            trace.append({"type": "tool_call", "tool": plan.tool_name, "arguments": args})
            trace.append({"type": "tool_result", "tool": plan.tool_name, "result": result})
            self.db.add_event(session_id, "tool_execution", {"tool": plan.tool_name, "result": result})

            # Auto-update active session entities (invoice, bc, product, doc, client)
            self._update_session_entities_from_result(session_id, plan.tool_name, args, result)

            reply, was_llm = self._synthesize_reply(
                user_text=text,
                tool_name=plan.tool_name,
                args=args,
                result=result,
                context=context,
            )
            if was_llm:
                trace.append({"type": "thought", "summary": f"Formulated response with LLM from tool '{plan.tool_name}' result."})

            active_doc_id = None
            doc_payload = None
            if isinstance(result, dict) and "document" in result and result["document"]:
                doc_payload = result["document"]
                active_doc_id = doc_payload.get("doc_id")
            elif isinstance(result, dict) and "doc_id" in result and result["doc_id"]:
                active_doc_id = result["doc_id"]
                doc_payload = self.db.get_document(active_doc_id)
            elif doc_id:
                active_doc_id = doc_id
                doc_payload = self.db.get_document(active_doc_id)

            file_url, file_name = self._extract_file_export_from_result(result)
            self.db.add_message(session_id, user_id, "assistant", reply)
            return {
                "session_id": session_id,
                "reply": reply,
                "status": "completed",
                "approval_token": None,
                "trace": trace,
                "doc_id": active_doc_id,
                "document": doc_payload,
                "file_url": file_url,
                "file_name": file_name,
            }

        reply = plan.reply or "Comment puis-je vous aider avec l'ERP aujourd'hui ?"
        self.db.add_message(session_id, user_id, "assistant", reply)
        active_doc_id = doc_id
        target_doc = self.db.get_document(active_doc_id) if active_doc_id else None
        return {
            "session_id": session_id,
            "reply": reply,
            "status": "completed",
            "approval_token": None,
            "trace": trace,
            "doc_id": active_doc_id,
            "document": target_doc,
            "file_url": None,
            "file_name": None,
        }

    def _update_session_entities_from_result(
        self,
        session_id: str,
        tool_name: str,
        args: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        """Extracts and updates active entity state (invoices, orders, products, documents, clients) in the session."""
        updates: dict[str, Any] = {}
        if isinstance(result, dict):
            if result.get("invoice_id"):
                updates["active_invoice_id"] = result["invoice_id"]
            elif args.get("invoice_id"):
                updates["active_invoice_id"] = args["invoice_id"]

            if result.get("order_id"):
                updates["active_bc_id"] = result["order_id"]
            elif args.get("order_id"):
                updates["active_bc_id"] = args["order_id"]

            if result.get("product_id"):
                updates["active_product_id"] = result["product_id"]
            elif isinstance(result.get("product"), dict) and result["product"].get("product_id"):
                updates["active_product_id"] = result["product"]["product_id"]
            elif args.get("product_id"):
                updates["active_product_id"] = args["product_id"]

            if result.get("doc_id"):
                updates["active_doc_id"] = result["doc_id"]
            elif isinstance(result.get("document"), dict) and result["document"].get("doc_id"):
                updates["active_doc_id"] = result["document"]["doc_id"]
            elif args.get("doc_id"):
                updates["active_doc_id"] = args["doc_id"]

            if result.get("client_name"):
                updates["active_client"] = result["client_name"]
            elif result.get("customer_id"):
                updates["active_client"] = result["customer_id"]
            elif args.get("client_name"):
                updates["active_client"] = args["client_name"]
            elif args.get("customer_id"):
                updates["active_client"] = args["customer_id"]

        if updates:
            self.db.update_session_state(session_id, **updates)

    def _extract_file_export_from_result(self, result: Any) -> tuple[str | None, str | None]:
        """Extracts relative download URL and clean filename from export tool results."""
        from pathlib import Path
        if not isinstance(result, dict):
            return None, None
        if result.get("download_url"):
            fn = result.get("filename") or (Path(result["pdf_path"]).name if result.get("pdf_path") else "document.pdf")
            return result["download_url"], fn
        if result.get("pdf_path"):
            p = Path(result["pdf_path"])
            return f"/v1/exports/download/{p.name}", p.name
        return None, None

    def _synthesize_reply(
        self,
        user_text: str,
        tool_name: str,
        args: dict[str, Any],
        result: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> tuple[str, bool]:
        """Uses Ollama to formulate a natural, contextual answer based on tool execution, falling back to deterministic formatting."""
        if self.settings.planner == "ollama":
            url = self.settings.ollama_base_url.rstrip("/") + "/api/chat"
            system = (
                "You are an expert ERP assistant. The user asked a question or requested an action, "
                "and a backend ERP tool was executed to process it.\n"
                "Your job is to formulate a clear, helpful, natural, and direct response to the user.\n"
                "- Match the user's language (French, English, or Arabic).\n"
                "- Clearly summarize what was done, key figures (TTC, HT, TVA, stock, quantities), dates, or items.\n"
                "- Use clean markdown formatting (bold, bullet points) when listing items or financial summaries.\n"
                "- Do NOT output raw JSON, tool call syntax, or code blocks unless explicitly requested.\n"
                "- If the tool returned an error or item not found, explain gently and suggest next steps."
            )
            user_msg = (
                f"User Request: {user_text}\n\n"
                f"Tool Executed: {tool_name}\n"
                f"Tool Arguments: {json.dumps(args, ensure_ascii=False)}\n"
                f"Execution Result:\n{json.dumps(result, ensure_ascii=False, default=str)}"
            )
            messages_payload: list[dict[str, str]] = [{"role": "system", "content": system}]
            if context and "messages" in context:
                for m in context["messages"][-4:]:
                    messages_payload.append({"role": m.get("role", "user"), "content": str(m.get("content", ""))})
            messages_payload.append({"role": "user", "content": user_msg})

            payload = {
                "model": self.settings.ollama_model,
                "think": False,
                "stream": False,
                "keep_alive": -1,
                "options": {"temperature": 0.2},
                "messages": messages_payload,
            }
            try:
                import httpx
                resp = httpx.post(
                    url,
                    json=payload,
                    headers={"ngrok-skip-browser-warning": "1", "User-Agent": "ERP-Agent/1.0"},
                    timeout=min(self.settings.ollama_timeout, 45.0),
                )
                if resp.status_code == 200:
                    content = resp.json().get("message", {}).get("content", "").strip()
                    if content:
                        return content, True
            except Exception:
                pass

        # Deterministic fallback formatter
        return self._format_deterministic_reply(tool_name, args, result), False

    def _format_deterministic_reply(
        self,
        tool_name: str,
        args: dict[str, Any],
        result: dict[str, Any],
    ) -> str:
        """Deterministic rule-based reply formatting as a solid fallback."""
        if tool_name in ("get_inventory", "get_product"):
            if result.get("found") and result.get("product"):
                prod = result["product"]
                return (
                    f"📦 **Produit {prod.get('name')} ({prod.get('product_id')})**\n\n"
                    f"• **Stock disponible :** {prod.get('stock')} unités\n"
                    f"• **Prix unitaire :** {prod.get('price')} TND"
                )
            return f"Aucun produit trouvé avec l'identifiant {args.get('product_id')}."

        if tool_name == "search_products":
            products = result.get("products", [])
            if products:
                items = "\n".join(f"• **{p['product_id']}** : {p['name']} — {p['stock']} en stock @ {p['price']} TND" for p in products)
                return f"🔍 **Produits trouvés ({len(products)}) :**\n{items}"
            return f"Aucun produit ne correspond à votre recherche '{args.get('query')}'."

        if tool_name == "create_product":
            if result.get("created"):
                p = result.get("product", {})
                return f"✅ Produit **{p.get('name')}** ({p.get('product_id')}) créé avec succès (Stock: {p.get('stock')}, Prix: {p.get('price')} TND)."
            return f"Échec de création du produit : {result.get('error', 'Erreur inconnue')}"

        if tool_name == "update_product_stock":
            if result.get("updated"):
                p = result.get("product", {})
                return f"✅ Produit **{p.get('product_id')}** mis à jour avec succès (Stock: {p.get('stock')} unités, Prix: {p.get('price')} TND)."
            return f"Échec de mise à jour du produit : {result.get('error', 'Erreur inconnue')}"

        if tool_name == "get_document":
            if result.get("found") and result.get("document"):
                doc = result["document"]
                return (
                    f"📄 **[{doc['doc_id']}] {doc['title']}** | Type: `{doc['doc_type']}` | Statut: `{doc['status']}`\n\n"
                    f"{doc['content']}"
                )
            return f"Document introuvable avec l'identifiant {args.get('doc_id')}."

        if tool_name == "search_documents":
            docs = result.get("documents", [])
            if docs:
                items = "\n".join(f"• **{d['doc_id']}** : {d['title']} (`{d['doc_type']}`, `{d['status']}`)" for d in docs)
                return f"📄 **Documents trouvés ({len(docs)}) :**\n{items}"
            return f"Aucun document ne correspond à votre recherche pour '{args.get('query')}'."

        if tool_name == "list_invoices":
            invoices = result.get("invoices", [])
            if invoices:
                items = "\n".join(
                    f"• **{inv['invoice_id']}** : {inv['client_name']} — **{inv['total_ttc']} {inv['currency']}** TTC ({inv['status']})"
                    for inv in invoices
                )
                return f"🧾 **Factures enregistrées ({len(invoices)}) :**\n\n{items}"
            return "Aucune facture enregistrée dans le système."

        if tool_name == "list_bon_de_commandes":
            orders = result.get("orders", [])
            if orders:
                items = "\n".join(
                    f"• **{o['order_id']}** : Client: {o['client_name']} — **{o['total_ttc']} TND** TTC ({o['status']})"
                    for o in orders
                )
                return f"📊 **Bons de Commande enregistrés ({len(orders)}) :**\n\n{items}"
            return "Aucun bon de commande enregistré."

        if tool_name == "delete_invoice":
            if result.get("deleted"):
                return f"🗑️ Facture **{result.get('invoice_id')}** supprimée avec succès."
            return f"Impossible de supprimer la facture : {result.get('error', 'Erreur')}"

        if tool_name == "duplicate_invoice":
            if result.get("found"):
                return result.get("message", f"Facture dupliquée avec succès vers {result.get('invoice_id')}.")
            return f"Échec de duplication : {result.get('error', 'Erreur')}"

        if tool_name == "export_invoice_pdf":
            if result.get("found"):
                return f"📄 Facture **{result.get('invoice_id')}** exportée avec succès en PDF.\nVous pouvez télécharger le fichier ci-dessous."
            return f"Échec de l'export PDF : {result.get('error', 'Erreur')}"

        if tool_name == "get_order_summary":
            if result.get("found"):
                fin = result.get("financials", {})
                items_txt = "\n".join(
                    f"• {it.get('name')} x{it.get('quantity')} @ {it.get('unit_price')} TND (-{it.get('discount_pct')}%): {it.get('line_net_ht')} TND HT"
                    for it in fin.get("items", [])
                ) or "Aucun article."
                return (
                    f"📊 **Bon de Commande : {result.get('order_id')}** (Client: {result.get('client_name')})\n\n"
                    f"**Articles :**\n{items_txt}\n\n"
                    f"• Total Brut HT : {fin.get('total_brut_ht')} TND\n"
                    f"• Total Remises : {fin.get('total_remises')} TND\n"
                    f"• Total Net HT : {fin.get('total_net_ht')} TND\n"
                    f"• TVA ({fin.get('tax_rate')}%) : {fin.get('total_tva')} TND\n"
                    f"• **TOTAL TTC : {fin.get('total_ttc')} TND**"
                )
            return f"Impossible de récupérer le bon de commande : {result.get('error', 'Erreur')}"

        if tool_name == "get_invoice_summary":
            if result.get("found"):
                fin = result.get("financials", {})
                cur = fin.get("currency", "TND")
                items_txt = "\n".join(
                    f"• {it.get('name')} x{it.get('quantity')} @ {it.get('unit_price')} {cur} (-{it.get('discount_pct')}%): {it.get('line_net_ht')} {cur} HT"
                    for it in fin.get("items", [])
                ) or "Aucun article."
                timbre_txt = f"\n• Timbre Fiscal : {fin.get('timbre_fiscal')} {cur}" if fin.get("timbre_fiscal") else ""
                return (
                    f"🧾 **Facture : {result.get('invoice_id')}** (Client: {result.get('client_name')} | Statut: {result.get('status')})\n\n"
                    f"**Articles & Prestations :**\n{items_txt}\n\n"
                    f"• Total Brut HT : {fin.get('total_brut_ht')} {cur}\n"
                    f"• Total Remises : {fin.get('total_remises')} {cur}\n"
                    f"• Total Net HT : {fin.get('total_net_ht')} {cur}\n"
                    f"• TVA ({fin.get('tax_rate')}%) : {fin.get('total_tva')} {cur}{timbre_txt}\n"
                    f"• **TOTAL TTC : {fin.get('total_ttc')} {cur}**"
                )
            return f"Impossible de récupérer la facture : {result.get('error', 'Erreur inconnue')}"

        if tool_name == "read_pdf":
            if result.get("found"):
                pages = result.get("pages_read", 0)
                total = result.get("total_pages", pages)
                sample = result.get("text", "")
                return f"PDF Content ({pages}/{total} pages read):\n\n{sample}"
            return f"Unable to read PDF: {result.get('error', 'Unknown error')}"

        if tool_name == "extract_pdf_tables":
            if result.get("found"):
                count = result.get("table_count", 0)
                tables = result.get("tables", [])
                return f"Extracted {count} tables from PDF:\n{json.dumps(tables, indent=2, ensure_ascii=False)}"
            return f"Table extraction failed: {result.get('error', 'Unknown error')}"

        if tool_name == "export_document_to_pdf":
            if result.get("found"):
                return f"📄 Document **{result.get('doc_id', '')}** exporté avec succès en PDF.\nVous pouvez télécharger le fichier ci-dessous."
            return f"PDF export failed: {result.get('error', 'Unknown error')}"

        if tool_name == "merge_pdfs":
            if result.get("found"):
                return result.get("message", f"PDFs merged successfully into {result.get('output_path')}")
            return f"PDF merge failed: {result.get('error', 'Unknown error')}"

        if isinstance(result, dict) and result.get("message"):
            return result["message"]
        if isinstance(result, dict) and not result.get("found", True) and result.get("error"):
            return f"Erreur lors de l'exécution de '{tool_name}' : {result['error']}"
        return f"Action '{tool_name}' effectuée avec succès."

    def approve(self, token: str) -> dict[str, Any]:
        appr = self.db.claim_approval(token)
        if not appr:
            existing = self.db.get_approval(token)
            if existing:
                return {
                    "approval_token": token,
                    "status": "error",
                    "result": None,
                    "message": f"Approval token is already {existing['status']}",
                    "file_url": None,
                    "file_name": None,
                }
            return {
                "approval_token": token,
                "status": "error",
                "result": None,
                "message": "Approval token not found",
                "file_url": None,
                "file_name": None,
            }

        try:
            tool = self.tools.get(appr["tool_name"])
            result = tool.handler(**appr["arguments"], user_id=appr["user_id"])
            self.db.finish_approval(token, "completed")
            self.db.add_event(
                appr["session_id"],
                "approval_completed",
                {"token": token, "tool_name": appr["tool_name"], "result": result},
            )

            # Auto-update active session entities
            self._update_session_entities_from_result(appr["session_id"], appr["tool_name"], appr["arguments"], result)

            context = self.memory.context(appr["session_id"], appr["user_id"])
            msg_text, _ = self._synthesize_reply(
                user_text=f"Confirmed action: {appr['tool_name']}",
                tool_name=appr["tool_name"],
                args=appr["arguments"],
                result=result,
                context=context,
            )

            self.db.add_message(
                appr["session_id"],
                appr["user_id"],
                "assistant",
                msg_text,
            )

            active_doc_id = None
            doc_payload = None
            if isinstance(result, dict) and "document" in result and result["document"]:
                doc_payload = result["document"]
                active_doc_id = doc_payload.get("doc_id")
            elif "doc_id" in appr.get("arguments", {}):
                active_doc_id = appr["arguments"]["doc_id"]
                doc_payload = self.db.get_document(active_doc_id)

            file_url, file_name = self._extract_file_export_from_result(result)

            return {
                "approval_token": token,
                "status": "completed",
                "result": result,
                "message": msg_text,
                "doc_id": active_doc_id,
                "document": doc_payload,
                "file_url": file_url,
                "file_name": file_name,
            }
        except Exception as exc:
            self.db.finish_approval(token, "failed")
            self.db.add_event(appr["session_id"], "approval_failed", {"token": token, "error": str(exc)})
            return {
                "approval_token": token,
                "status": "error",
                "result": None,
                "message": f"Execution failed: {exc}",
                "file_url": None,
                "file_name": None,
            }
