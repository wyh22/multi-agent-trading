# Company Research Skill

## Purpose
Analyze a listed company using financial data, official disclosures and retrieved evidence.

## When To Use
- User asks about a specific stock.
- User requests company fundamentals or business analysis.

## Inputs
- ticker: stock symbol
- as_of_date: research cutoff date

## Required Tools
- fundamentals
- news
- rag_search

## Workflow
1. Retrieve company context.
2. Collect historical financial information.
3. Retrieve disclosure evidence.
4. Generate an evidence-backed summary.

## Constraints
- Do not provide investment advice.
- Do not use information after the research cutoff date.
- Separate facts, calculations and inference.
