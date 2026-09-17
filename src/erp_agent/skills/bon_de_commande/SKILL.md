---
name: bon_de_commande
description: Use this skill whenever the user wants to create, manage, edit, check totals, add/remove items, or export a "Bon de Commande" (Purchase Order / Sales Order / BC / طلب شراء / طلبية). Supports line additions, quantity updates, unit prices, discount rates (remises), VAT (TVA) calculations, subtotal HT, total TTC, and PDF generation.
license: Proprietary
---

# Bon de Commande & Financial Management Guide

## Overview

This skill provides financial tools for handling **Bons de Commande** (Purchase Orders / Sales Orders / BC) in the ERP system. It automatically calculates subtotals, item discounts, applicable VAT (TVA), and net total (TTC).

## Financial Formulas

1. **Line Subtotal Brut:**
   $$\text{Brut HT} = \text{Quantity} \times \text{Unit Price}$$

2. **Line Remise / Discount:**
   $$\text{Remise Amount} = \text{Brut HT} \times \left(\frac{\text{Discount \%}}{100}\right)$$

3. **Line Net HT:**
   $$\text{Line Net HT} = \text{Brut HT} - \text{Remise Amount}$$

4. **Document Totals:**
   - **Total Brut HT** $= \sum \text{Brut HT}$
   - **Total Remises** $= \sum \text{Remise Amount}$
   - **Total Net HT** $= \text{Total Brut HT} - \text{Total Remises}$
   - **Total TVA (e.g. 19%)** $= \text{Total Net HT} \times \left(\frac{\text{Tax Rate \%}}{100}\right)$
   - **Total TTC** $= \text{Total Net HT} + \text{Total TVA}$

## Available Tools

- `create_bon_de_commande(client_name, items, tax_rate, discount_pct, order_id, currency)`: Initializes a new Bon de Commande with client info and optional items. (Requires confirmation).
- `add_order_item(order_id, product_id, name, quantity, unit_price, discount_pct)`: Adds a new line item to an existing order. Automatically fetches catalog price and stock if `product_id` is provided. (Requires confirmation).
- `update_order_item(order_id, item_index, product_id, quantity, unit_price, discount_pct)`: Updates an existing line item. (Requires confirmation).
- `remove_order_item(order_id, item_index, product_id)`: Removes a line item. (Requires confirmation).
- `get_order_summary(order_id)`: Computes and returns the complete itemized breakdown and financial summary (Total HT, Remises, TVA, Total TTC). Safe read action.
- `export_bon_de_commande_pdf(order_id, output_path)`: Exports the complete Bon de Commande as a formal PDF document. (Requires confirmation).

## Usage Examples

- "Crée un bon de commande pour la société Alpha avec 2x Laptop Pro et 5x Wireless Mouse"
- "Ajoute 3x USB-C Dock au bon de commande BC-101 avec 10% de remise"
- "Modifie la quantité du Laptop Pro dans BC-101 à 4 unités"
- "Supprime la souris du bon de commande BC-101"
- "Calcule le total TTC et la TVA du bon de commande BC-101"
- "Exporte le bon de commande BC-101 en PDF"
