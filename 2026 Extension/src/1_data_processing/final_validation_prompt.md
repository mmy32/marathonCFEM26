# Task: Final validation of borrower, seniority and sector labels for BDC investments

You are validating three labels attached to investment positions held by US business development companies (BDCs). The positions come from the schedules of investments in BDC 10-K and 10-Q iXBRL filings, 2023Q1–2026Q2. Each position is identified by `investment_identifier`, the free-text label the BDC gave it in its filing.

The three labels were produced earlier:

- `borrower_canonical`: a canonical borrower name from a reviewed lookup table, written in a fixed format (see section 1), e.g. `PROJECT ALPHA INTERMEDIATE HOLDING`. It is used to link the same borrower across quarters and across BDCs, so its spelling matters.
- `instrument_seniority`: keyword rules applied to the identifier.
- `sector`: keyword rules applied to the identifier. Where the rules found nothing, an earlier LLM pass chose a sector, using general knowledge of the company, if its confidence was at least 0.70.

Your job is to confirm, correct or fill each label. The aim is a conservative, auditable result: **accuracy takes precedence over completeness.** Never force a value when the evidence is insufficient.

## Input

Each request contains one batch: a JSON array of objects, one per unique identifier.

```json
{"id": 17, "investment_identifier": "...", "borrower_canonical": "...", "instrument_seniority": "...", "sector": "..."}
```

`borrower_canonical` may be `""` (not covered by the lookup). A batch may also include `"known_borrowers"`: canonical names already used for the same BDC. Identifiers in a batch are grouped by BDC, so several rows often describe the same borrower.

## Reading an identifier

Identifiers mix several kinds of text, often in one string:

- portfolio headings: "Investments - non-controlled/non-affiliated", "Debt Investments";
- an industry label reported by the BDC for the borrower: "QC Holdings TopCo, LLC | **Consumer Finance** | Second Lien Term Loan";
- the borrower's name, sometimes with "dba", "f/k/a" or co-borrowers;
- the instrument: "First lien senior secured loan", "Revolver", "Series B Preferred Stock";
- terms: spread, coupon, PIK, floor, maturity, footnote markers such as "(11)" or a trailing digit.

Some identifiers are XBRL member names written in CamelCase ("ConvertibleNoteIntraopMedicalCorpMedicalDevicesSectorNoteOneMember"), and some are not positions at all (e.g. "1 Month SOFR Member", a reference rate). Words that look like instrument terms can be part of a company name ("One Stop Mailing"). Read the whole string before deciding.

Use only the identifier text, the current labels and the other identifiers in the same batch as evidence. The single exception is the sector rule in section 3.

## 1. `borrower_canonical`

The borrower is the portfolio company or issuer, not the BDC, a fund heading, an instrument or a reference rate.

- **Keep the current value** unless:
  - the identifier clearly names a different entity;
  - the value is an instrument, heading or rate rather than an entity;
  - the value is a lone generic word left by truncation (`BIDCO`, `TOPCO`, `BUYER`, `HOLDCO`, `PARENT`, `INTERMEDIATE`, `ACQUISITION`).
- **Reuse existing names.** When filling or correcting a value, use the exact name from `known_borrowers`, or from another row in the batch for the same entity, whenever the entity matches. Only write a new name when no existing one fits.
- **New names** must follow the dataset's format:
  1. Take the legal name as written, minus instrument, terms, footnote markers and "dba" aliases. Split CamelCase member names into words.
  2. Write it in upper case without accents.
  3. Delete periods and apostrophes (`J.P.` → `JP`); turn all other punctuation, including `&` and `-`, into single spaces.
  4. Drop legal-form words: LLC, Inc, Incorporated, Corp, Corporation, Ltd, LP, LLP, PLC, GmbH, S.à r.l./SARL and ULC wherever they follow the first word. Drop Co, Company, Limited, S.A., SAS, S.p.A., AG, KG, AB, B.V., N.V., Pty, DAC and spaced forms like "L L C" only when they come last.
  5. Keep every other word. HOLDINGS, BUYER, PARENT, BIDCO, TOPCO, II and similar words distinguish separate legal entities.

  Example: "Sensor Technology Topco, Inc. (dba Humanetics)" → `SENSOR TECHNOLOGY TOPCO`.
- **Do not merge distinct entities.** A subsidiary, co-borrower or similarly named company is not the same borrower unless the identifier says so.
- **Missing value:** `""`.

## 2. `instrument_seniority`

Use exactly one of these codes:

| Code | Use when the identifier states |
|---|---|
| `SENIOR_SECURED_1L` | first lien / 1st lien, including "first lien senior secured" |
| `SENIOR_SECURED_2L` | second lien / 2nd lien, or junior secured |
| `SENIOR_SECURED_UNSPEC` | secured or senior secured, with no lien position given |
| `UNITRANCHE` | unitranche, or a first-out/last-out split (FO/LO) |
| `SENIOR_UNSECURED` | unsecured, or senior notes / senior debt not described as secured |
| `SUBORDINATED_MEZZ` | subordinated, mezzanine, junior debt or notes, HoldCo notes, loans or debt, PIK toggle notes |
| `CLO_STRUCTURED_EQUITY` | CLO equity or subordinated notes, collateralized loan obligation, structured finance membership interests |
| `OTHER_UNKNOWN` | anything else: equity (preferred, common, warrants, units), an instrument whose seniority is not stated, non-positions, and any case below 0.70 confidence |

Rules:

- **Not enough on its own:** "loan", "term loan", "revolver", "delayed draw", a coupon or a maturity is not evidence of seniority. "Sicilian Oven Restaurants LLC – Term Debt" is `OTHER_UNKNOWN`.
- **Lien position is not payment priority.** Do not promote a secured instrument to `SENIOR_SECURED_1L` without a stated first lien.
- **Lender-specific terms** may be used only when they clearly describe the instrument. For example, Golub Capital labels unitranche loans "One stop". Never use such a term when it is part of a company name.
- **More than one category:** if the identifier contains descriptors for more than one category, prefer the more specific in this order: `CLO_STRUCTURED_EQUITY`, `SUBORDINATED_MEZZ`, `UNITRANCHE`, `SENIOR_SECURED_2L`, `SENIOR_UNSECURED`, `SENIOR_SECURED_1L`, `SENIOR_SECURED_UNSPEC`.
- **Missing value:** `OTHER_UNKNOWN`.

## 3. `sector`

Use exactly one of these labels:

| Label | Scope |
|---|---|
| `Healthcare` | providers, clinics, hospitals, dental, vision, imaging, fertility; pharma, biotech, life sciences; medical devices; pharmacy |
| `Technology` | software and SaaS, IT services, cybersecurity, cloud, data centers, digital platforms |
| `Financials & Insurance` | banks, consumer and specialty finance, lenders, funds and asset managers, insurance and insurance brokerage, CLOs and structured finance |
| `Transportation` | transport operators: airlines, shipping and marine transport, fleet services |
| `Consumer` | food, beverage, tobacco, grocery; restaurants, hotels, leisure, gaming, fitness; retail and wholesale of consumer goods; education and training; consumer services |
| `Industrials` | manufacturing, engineering, equipment; logistics, distribution, warehousing, trucking, freight, rail; commercial and professional services |
| `Energy & Utilities` | oil and gas, pipelines, power, renewables, electric, gas and water utilities |
| `Materials` | chemicals, plastics, rubber, packaging, other basic materials |
| `Real Estate` | property owners and operators, REITs, apartments, office, industrial and logistics parks |
| `Communication & Media` | telecom, cable, wireless carriers, media, advertising and marketing agencies, entertainment, streaming, studios, music |
| `Other / Unknown` | the business is identified but fits none of the ten sectors above |
| `Other / Unknown / Unresolved` | the business cannot be identified reliably, or confidence is below 0.70 |

Where the scope notes are silent, follow GICS conventions (Consumer Discretionary and Consumer Staples → `Consumer`; Information Technology → `Technology`; Communication Services → `Communication & Media`). Note that "Consumer Finance" is `Financials & Insurance`.

Evidence, strongest first:

1. **The industry label the BDC reports for the borrower** (e.g. "| Consumer Finance |", "- IT Consulting -").
2. **Business words in the identifier or borrower name** (e.g. "Restaurants", "Pharmaceuticals"). Do not infer a sector from a name that is ambiguous.
3. **General knowledge of the specific, unambiguously identified company.** This is permitted for `sector` only, and only when you would be at least 0.70 confident. Label it `general_knowledge` in the output so it can be filtered later.

Not evidence of sector:

- financing words (loan, first lien, revolver, facility, credit agreement);
- portfolio headings ("non-controlled/non-affiliated", "Debt Investments");
- the fact that the holder is a BDC.

All rows for the same borrower in a batch should get the same sector. Use the strongest evidence among them: if one QC Holdings row reports "Consumer Finance", a sibling row without the label is also `Financials & Insurance`, not a guessed sector.

Missing value: `Other / Unknown / Unresolved`.

## 4. Confidence and changes

- **Score each field independently** from 0.00 to 1.00, by strength of evidence. This is a disciplined assessment, not a calibrated probability.
- **Keep the current value** when it is supported at 0.70 or above. Do not rewrite supported values for style.
- **Replace it** only when the current value is wrong and your replacement reaches 0.70.
- **Otherwise** output the field's missing value: `""`, `OTHER_UNKNOWN` or `Other / Unknown / Unresolved`. Never keep or impute a value below 0.70 just to fill the field.

## 5. Output

Return only a JSON array, with one object per input object, in the same order and with the same `id`. Do not echo `investment_identifier`.

```json
{"id": 17,
 "borrower_canonical": "QC HOLDINGS TOPCO", "borrower_confidence": 0.95,
 "instrument_seniority": "SENIOR_SECURED_2L", "seniority_confidence": 0.95, "seniority_evidence": "Second Lien Term Loan",
 "sector": "Financials & Insurance", "sector_confidence": 0.85, "sector_evidence_type": "reported_industry", "sector_evidence": "Consumer Finance (sibling row)"}
```

- `sector_evidence_type` is one of `reported_industry`, `identifier_text`, `general_knowledge` or `none`.
- The two `*_evidence` fields hold a quote of at most 8 words from the identifier, or a brief note; use `""` when there is none.
- Labels must match the lists above exactly, including case and spacing.
- Write no prose outside the JSON.

Before returning, check that:

- the array has as many objects as the input, in the same `id` order;
- every label is from the allowed lists;
- no value below 0.70 has been kept or imputed.
