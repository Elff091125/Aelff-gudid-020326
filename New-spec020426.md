Technical Specification — WOW UI Redesign & Multi‑Dataset Search/Visualization Studio (Streamlit on Hugging Face Spaces)

> **Constraint:** This document specifies *design and technical behavior only*. It does **not** modify or generate code. All existing features in `app.py` must remain available. New capabilities are additive and must integrate with the existing navigation, session state patterns, and dataset/agent tooling already present.

---

## 1) Product Goals & Scope

### 1.1 Goals
1. **Create a new “WOW UI” Search Studio** where a user enters a keyword once and the system searches **four datasets simultaneously**:
   - **510(k)** (premarket notifications)
   - **Recall**
   - **ADR** (adverse event / MDR-like)
   - **GUDID** (UDI/device identifier records)

2. **Present results separately per dataset** (four panels) **and** highlight cross-dataset relationships in *coral color* (consistent with `CORAL = #FF7F50`).

3. Provide **interactive visualization** and **drill-down**: users can click charts and records to see further information, linked entities, and suggested next searches.

4. Allow users to **keep prompts on results** and run agents from `agents.yaml` to do **visualization and analysis** on selected subsets/results.

5. Preserve the existing “WOW UI” concept (themes, language, painter styles), **status indicators**, and existing pages:
   - Dashboard, Command Center, Dataset Studio, Agent Studio, Factory, AI Note Keeper  
   The new Search Studio can be introduced as a new page or as a major enhancement to the existing Dashboard/Command Center search area—without removing existing routing options.

### 1.2 Non-Goals
- No new backend services beyond what is already used (Streamlit + APIs).
- No requirement to fetch real FDA datasets; support continues for mock/default datasets plus user-uploaded datasets.

---

## 2) Platform & Deployment Constraints (Hugging Face Spaces + Streamlit)

### 2.1 Runtime Environment
- Deployed on **Hugging Face Spaces** using **Streamlit**.
- Files used:
  - `app.py` (main UI)
  - `defaultsets.json` (default mock datasets)
  - `agents.yaml` (agent definitions)
  - `SKILL.md` (global constraints/instructions shared across agents)

### 2.2 Supported LLM APIs (per current request)
- **OPENAI API**
- **GEMINI API**
- Models selectable (minimum for agent execution in doc/results workflows):
  - `gpt-4o-mini`
  - `gemini-2.5-flash`
  - `gemini-3-flash-preview`

> Note: `app.py` currently includes other providers; this spec requires at least OpenAI+Gemini support for the new features while keeping existing provider logic intact.

### 2.3 API Key Handling Requirements
- If an API key is present in the environment (HF Space Secrets), the UI must show **“Authenticated via Environment”** and **must not display** or request the key input.
- If missing from environment, allow user input **in-page**; store only in **session state**.
- Provide clear “missing” indicators in the status area.

---

## 3) WOW UI System: Global UX Foundations

### 3.1 Themes, Language, Painter Styles (Keep + Extend)
Preserve the existing:
- Light/Dark theme selector
- English / Traditional Chinese selector
- 20 painter styles with a “Jackpot” randomizer

**New requirement for Search Studio:** The painter accent color must propagate into:
- Chart color palettes (primary accent)
- Selected record highlight background (subtle)
- Relationship edges/nodes (accent + coral emphasis)

### 3.2 WOW Status Indicators (Enhance)
Keep the current top “chips” concept and expand into a **status strip** that includes:
- API status per provider (OpenAI/Gemini at minimum)
- Dataset loaded counts per dataset
- Index/search readiness (e.g., “Search index: READY” once datasets exist)
- OCR readiness (if doc features are used)
- Agent config readiness (“agents.yaml: OK / Needs standardization”)

**Behavior:**
- Status chips update immediately on dataset changes, YAML updates, or OCR completion.
- Clicking a chip opens a compact popover with “what this means” + quick actions (e.g., “Load defaults”, “Open Dataset Studio”, “Upload agents.yaml”).

---

## 4) Information Architecture (Pages & Navigation)

### 4.1 New or Enhanced Page: “WOW Search Studio”
A dedicated page (recommended) focused on:
- **Single query input**
- **Parallel search across four datasets**
- **Separated results + linked relationship view**
- **Interactive filters + suggestions**
- **Agent-driven analysis & visualization**

If adding a new page is not desired, implement as the primary expanded experience within “Dashboard” while keeping existing Dashboard content accessible via tabs.

### 4.2 Recommended Layout (Wide Mode)
**Top:** Global command bar (query + search settings) stays consistent.

**Main content split (3-tier):**
1. **Row A: KPI + Quick Insights**
   - Total hits
   - Hits by dataset
   - Top linked manufacturer/product_code/UDI
   - “Relationship density” indicator (how interconnected results are)

2. **Row B: Four Dataset Panels (separate)**
   - 510(k) panel
   - Recall panel
   - ADR panel
   - GUDID panel  
   Each panel contains:
   - Mini charts (timeline + distribution)
   - Result table
   - Click-to-open detail drawer

3. **Row C: Relationship Explorer + Suggestions**
   - Graph/network view of entities connecting the four datasets
   - “Why linked?” explanation panel
   - Suggested next queries + facet filters

---

## 5) Multi-Dataset Search Engine: Behavior & Settings

### 5.1 Search Execution
- Single query `q` triggers search on:
  - `dfs["510k"]`, `dfs["recall"]`, `dfs["adr"]`, `dfs["gudid"]`
- Results stored per dataset as ranked lists with score.

### 5.2 Dataset Separation Requirement
The UI must show four distinct result sections, each with:
- Dataset name + hit count
- Dataset-specific filters
- Dataset-specific chart set
- Dataset-specific table

### 5.3 Search Settings (User‑Modifiable)
Search settings must be editable in a **Search Settings** expander/panel:
- Exact match toggle
- Fuzzy threshold slider
- Dataset toggles (include/exclude each dataset)
- Result limit per dataset (default 200)
- Field weighting mode:
  - Balanced (default)
  - ID-boosted (boost K#, recall number, UDI/DI)
  - Narrative-boosted (boost summary/narrative fields)
- Date range filters where applicable:
  - 510(k): `decision_date`
  - Recall: `event_date`, optionally `termination_date`
  - ADR: `report_date`
  - GUDID: `publish_date`

### 5.4 Query Understanding (Suggestions & Expansions)
After a search, the system generates:
- **Suggested next searches** (chips/buttons), derived from:
  - Top entities found (manufacturer_name, product_code, brand_name, UDI/DI)
  - Detected IDs (K-number patterns `K\d+`, recall numbers, DI formats)
  - Related terms from an ontology (existing coral highlight ontology can be reused)

**Examples:**
- “Search same manufacturer across all datasets”
- “Search product_code only”
- “Search linked recall numbers from ADR results”
- “Search predicates from top 510(k) result”

Suggestions appear:
- Globally (below results)
- Within each dataset panel (“Refine within Recall”)

---

## 6) Interactive Visualization Requirements (Per Dataset + Cross Dataset)

### 6.1 Common Interaction Model (All Charts)
All charts must support:
- Hover tooltips showing key identifiers and score
- Click or select to:
  - Filter the dataset table
  - Update the relationship explorer
  - Update the “detail drawer” context
- A “Reset filters” action per panel

### 6.2 Dataset-Specific Visualization Sets

#### 6.2.1 510(k) Panel
**Charts:**
- Timeline scatter/line by `decision_date` (size by score)
- Distribution by `panel` or `device_class` or `decision`
- Optional: product_code bar chart (top N)

**Table columns priority:**
- `k_number`, `device_name`, `applicant`, `manufacturer_name`, `product_code`, `decision_date`, `decision`, `device_class`

**Drill-down detail drawer tabs:**
- Overview (clean fields + highlighted keywords)
- Predicates (show `predicate_k_numbers` list and enable click-to-search)
- Agent actions (run a selected agent on this record + related records)
- Raw JSON (verbatim record)

#### 6.2.2 Recall Panel
**Charts:**
- Timeline by `event_date`
- Distribution by `recall_class` and `status`
- “Reason for recall” keyword cluster (lightweight: top terms extracted locally or via agent)

**Table columns priority:**
- `recall_number`, `recall_class`, `status`, `firm_name`, `manufacturer_name`, `product_description`, `product_code`, `reason_for_recall`, `event_date`

**Drill-down tabs:**
- Overview
- Distribution pattern + quantity
- Linked entities (UDI/product_code/manufacturer + linked ADR)
- Raw JSON

#### 6.2.3 ADR Panel
**Charts:**
- Timeline by `report_date`
- Distribution by `event_type` and `patient_outcome`
- Device problem bar chart (top N)

**Table columns priority:**
- `adverse_event_id`, `report_date`, `event_type`, `patient_outcome`, `device_problem`, `manufacturer_name`, `brand_name`, `product_code`, `udi_di`, `recall_number_link`

**Drill-down tabs:**
- Narrative (with coral highlights)
- Linked recall (if `recall_number_link` exists)
- Linked GUDID (if `udi_di` exists)
- Raw JSON

#### 6.2.4 GUDID Panel
**Charts:**
- Publish timeline by `publish_date`
- Distribution by `device_class`, `mri_safety`, `sterile/single_use/implantable`
- Manufacturer geography (if state/country exist)

**Table columns priority:**
- `primary_di` / `udi_di`, `device_description`, `manufacturer_name`, `brand_name`, `product_code`, `device_class`, `mri_safety`, `publish_date`

**Drill-down tabs:**
- Device identifiers (primary_di, udi_di)
- Safety flags (sterile/single_use/implantable/contains_nrl)
- Contact (email/phone if present)
- Raw JSON

---

## 7) Relationship Highlighting (Cross‑Dataset Linking in Coral)

### 7.1 Linking Objectives
The system must reveal relationships across the four datasets and highlight them in **coral** in both:
- Tables (cell-level highlights)
- Relationship Explorer (nodes/edges and “shared fields” badges)

### 7.2 Entity Types for Linking
Build relationships using these **entity keys** (ranked by reliability):

1. **UDI/DI**  
   - ADR: `udi_di`  
   - GUDID: `udi_di`, `primary_di`

2. **Recall number**  
   - Recall: `recall_number`  
   - ADR: `recall_number_link`

3. **Manufacturer / Firm** (normalized string)
   - 510(k): `manufacturer_name`, `applicant`
   - Recall: `manufacturer_name`, `firm_name`
   - ADR: `manufacturer_name`
   - GUDID: `manufacturer_name`

4. **Product code**
   - All datasets contain `product_code`

5. **Brand / Device name similarity**
   - ADR: `brand_name`
   - GUDID: `brand_name`
   - 510(k): `device_name`
   - Recall: `product_description`

### 7.3 Normalization Rules (Display + Linking)
To link reliably, define normalization (no code here; required behavior):
- Trim whitespace; collapse multiple spaces
- Casefold (lowercase for matching)
- Remove punctuation except meaningful separators for IDs
- Keep original value for display, but use normalized for linking

### 7.4 Relationship Explorer (Interactive Graph)
Provide a dedicated section that shows:
- **Entity nodes** (manufacturer, product_code, udi_di, recall_number, k_number)
- **Record nodes** (optional; or represent record counts aggregated)

**Edges:**
- Connect entity-to-record or entity-to-entity
- Edge thickness = number of records supporting the link
- Coral highlight used for:
  - Entities that appear in >=2 datasets
  - Edges that bridge dataset boundaries

**Interactions:**
- Clicking a node filters all four panels to linked records
- Clicking an edge shows “Why linked?” with:
  - Shared field name(s)
  - Count of supporting records
  - Confidence score category (High for exact ID match; Medium for product_code; Low for fuzzy name similarity)

### 7.5 “Relation Badges” Inside Each Dataset Panel
Within each dataset table, add a derived column (display-only):
- `linked_to`: a compact set of badges like:
  - `GUDID:UDI` (if UDI exists in GUDID hits)
  - `RECALL:#` (if recall number linked)
  - `ADR:signals` (if ADR mentions same manufacturer/product_code)

Badges that represent cross-dataset links must be coral or coral-bordered.

---

## 8) Drill‑Down: “Record Detail Drawer” Specification

When a user clicks a row in any dataset table:
- Open a **detail drawer** (right side panel or modal) containing:

### 8.1 Header
- Dataset name + primary identifier (coral)
- Match score + matched fields (explainability)
- Quick actions:
  - “Pin to Workspace”
  - “Add note”
  - “Run agent”
  - “Copy JSON”

### 8.2 Tabs
1. **Overview**: key fields rendered cleanly + coral highlights for ontology terms.
2. **Linked Records**: show related hits in other datasets with rationale (UDI, recall number, manufacturer, product_code).
3. **Search Refinement**: one-click chips to run new queries based on this record (e.g., click product_code).
4. **Agent Actions**: choose an agent and run it against:
   - This record only
   - All filtered results in this dataset
   - All linked results across datasets  
   User can edit prompt + select model before execution.
5. **Raw**: JSON view.

---

## 9) Agent-Driven Analysis & Visualization on Search Results

### 9.1 “Keep Prompt on Results”
Add a “Prompt Notebook” panel associated with each search session:
- Stores:
  - The query
  - Active filters
  - Selected records (pinned)
  - A user prompt text area (“What I want the agent to do”)
  - Model selection
- A “Run” button executes chosen agent(s) using:
  - `SKILL.md` + agent’s `system_prompt` + the user’s per-run prompt + a serialized view of the selected records.

### 9.2 Agent Execution Modes for Search Studio
Support these modes:
1. **Single Agent Run**: pick one agent and run once
2. **Pipeline**: run agents one-by-one (existing pipeline behavior), where:
   - User can edit each agent’s output (text/markdown)
   - The edited output becomes the next agent’s input
3. **Visualization Agent**: an agent that outputs:
   - Plot specifications (Markdown + JSON for chart configs) OR
   - A structured “insights report” that the UI can render

> Because this is Streamlit, “agent outputs chart instructions” can be rendered as Markdown and/or used as guidance to configure interactive charts; the system must keep outputs reproducible and auditable.

### 9.3 Model & Prompt Controls (Per Run)
Before execution:
- User can modify:
  - provider (OpenAI/Gemini)
  - model (`gpt-4o-mini`, `gemini-2.5-flash`, `gemini-3-flash-preview`)
  - max tokens (default 12000 where applicable; for search results summarization may default lower)
  - temperature
  - system prompt and user prompt

### 9.4 Safety & Non-Fabrication
Agents must:
- Quote evidence from records
- Mark missing info as **Gap**
- Avoid inventing regulatory conclusions

---

## 10) Dataset Search Refinement Tools (“Further Dig” Controls)

Each dataset panel must include a collapsible **Refine** section with dataset-specific filters:

### 10.1 510(k) Filters
- `decision_date` range
- `decision` (SESE / NSE / etc. as present)
- `device_class`
- `panel`
- `product_code`
- Applicant/manufacturer string filter

### 10.2 Recall Filters
- `event_date` range
- `recall_class`
- `status`
- `state/country`
- `product_code`
- Contains term filter for `reason_for_recall` and `product_description`

### 10.3 ADR Filters
- `report_date` range
- `event_type`, `patient_outcome`
- `device_problem`
- `product_code`
- Presence toggles:
  - has `udi_di`
  - has `recall_number_link`

### 10.4 GUDID Filters
- `publish_date` range
- `device_class`
- `mri_safety`
- booleans: `sterile`, `single_use`, `implantable`, `contains_nrl`
- `product_code`
- Manufacturer/brand filter

---

## 11) Mock Datasets (All 4 Included) — For Specification & UI Demonstration

These mock datasets must be included in `defaultsets.json` (or equivalent) to demonstrate cross-dataset linking. The UI must be able to search them and show relationships highlighted in coral.

> Below are **spec mock records** (example content). Actual default dataset may include more rows; these must exist at minimum to validate the relationship explorer.

### 11.1 Mock 510(k) Dataset (JSON records)
```json
[
  {
    "k_number": "K240123",
    "decision_date": "2024-06-18",
    "decision": "SESE",
    "device_name": "NovaPulse Infusion Pump",
    "applicant": "NovaMed Systems, Inc.",
    "manufacturer_name": "NovaMed Systems, Inc.",
    "product_code": "FRN",
    "regulation_number": "880.5725",
    "device_class": "II",
    "panel": "Anesthesiology",
    "review_advisory_committee": "—",
    "predicate_k_numbers": ["K221111", "K210987"],
    "summary": "Battery-powered infusion pump with wireless connectivity and dose error reduction software."
  },
  {
    "k_number": "K221111",
    "decision_date": "2022-04-10",
    "decision": "SESE",
    "device_name": "NovaPulse Infusion Pump (Gen1)",
    "applicant": "NovaMed Systems, Inc.",
    "manufacturer_name": "NovaMed Systems, Inc.",
    "product_code": "FRN",
    "regulation_number": "880.5725",
    "device_class": "II",
    "panel": "Anesthesiology",
    "review_advisory_committee": "—",
    "predicate_k_numbers": ["K190222"],
    "summary": "Earlier generation infusion pump; predicate chain used for substantial equivalence."
  },
  {
    "k_number": "K230456",
    "decision_date": "2023-11-02",
    "decision": "SESE",
    "device_name": "CardioSense ECG Patch",
    "applicant": "HeartArc Medical",
    "manufacturer_name": "HeartArc Medical",
    "product_code": "DXH",
    "regulation_number": "870.2800",
    "device_class": "II",
    "panel": "Cardiovascular",
    "review_advisory_committee": "—",
    "predicate_k_numbers": ["K201010"],
    "summary": "Single-use ECG monitoring patch. Includes mobile app data review."
  },
  {
    "k_number": "K231777",
    "decision_date": "2023-12-14",
    "decision": "SESE",
    "device_name": "OrthoAlign Surgical Navigation",
    "applicant": "OrthoWorks Ltd.",
    "manufacturer_name": "OrthoWorks Ltd.",
    "product_code": "HDD",
    "regulation_number": "888.1100",
    "device_class": "II",
    "panel": "Orthopedic",
    "review_advisory_committee": "—",
    "predicate_k_numbers": ["K210333"],
    "summary": "Navigation software used for orthopedic alignment; cybersecurity and software V&V referenced."
  }
]
```

### 11.2 Mock Recall Dataset
```json
[
  {
    "recall_number": "Z-1234-2024",
    "recall_class": "II",
    "event_date": "2024-08-05",
    "termination_date": "",
    "status": "Ongoing",
    "firm_name": "NovaMed Systems, Inc.",
    "manufacturer_name": "NovaMed Systems, Inc.",
    "product_description": "NovaPulse Infusion Pump",
    "product_code": "FRN",
    "code_info": "Lots: NP-2406A to NP-2407F",
    "reason_for_recall": "Potential battery failure leading to interruption of infusion.",
    "distribution_pattern": "US nationwide",
    "quantity_in_commerce": 3100,
    "country": "US",
    "state": "CA"
  },
  {
    "recall_number": "Z-0456-2023",
    "recall_class": "III",
    "event_date": "2023-03-20",
    "termination_date": "2023-10-01",
    "status": "Terminated",
    "firm_name": "HeartArc Medical",
    "manufacturer_name": "HeartArc Medical",
    "product_description": "CardioSense ECG Patch",
    "product_code": "DXH",
    "code_info": "Serials: CS-ECG-0001 to 2100",
    "reason_for_recall": "Labeling issue: contraindication not adequately displayed in IFU.",
    "distribution_pattern": "US limited distribution",
    "quantity_in_commerce": 850,
    "country": "US",
    "state": "MA"
  },
  {
    "recall_number": "Z-0999-2024",
    "recall_class": "I",
    "event_date": "2024-01-12",
    "termination_date": "",
    "status": "Ongoing",
    "firm_name": "OrthoWorks Ltd.",
    "manufacturer_name": "OrthoWorks Ltd.",
    "product_description": "OrthoAlign Surgical Navigation",
    "product_code": "HDD",
    "code_info": "Versions: 3.2.0 to 3.2.3",
    "reason_for_recall": "Software malfunction may display incorrect alignment guidance under rare conditions.",
    "distribution_pattern": "US and EU",
    "quantity_in_commerce": 120,
    "country": "US",
    "state": "NY"
  }
]
```

### 11.3 Mock ADR Dataset
```json
[
  {
    "adverse_event_id": "ADR-2024-000778",
    "report_date": "2024-08-22",
    "event_type": "Malfunction",
    "patient_outcome": "No injury",
    "device_problem": "Power/Battery Problem",
    "manufacturer_name": "NovaMed Systems, Inc.",
    "brand_name": "NovaPulse",
    "product_code": "FRN",
    "device_class": "II",
    "udi_di": "00810000012345",
    "recall_number_link": "Z-1234-2024",
    "narrative": "Unit shut down unexpectedly during infusion. Battery indicator showed adequate charge prior to event."
  },
  {
    "adverse_event_id": "ADR-2024-000812",
    "report_date": "2024-09-02",
    "event_type": "Serious Injury",
    "patient_outcome": "Hospitalization",
    "device_problem": "Dose Delivery Issue",
    "manufacturer_name": "NovaMed Systems, Inc.",
    "brand_name": "NovaPulse",
    "product_code": "FRN",
    "device_class": "II",
    "udi_di": "00810000012345",
    "recall_number_link": "Z-1234-2024",
    "narrative": "Reported interruption of therapy; investigation ongoing; possible power failure coincident with alarm."
  },
  {
    "adverse_event_id": "ADR-2023-000155",
    "report_date": "2023-04-09",
    "event_type": "Malfunction",
    "patient_outcome": "No injury",
    "device_problem": "Labeling/IFU Problem",
    "manufacturer_name": "HeartArc Medical",
    "brand_name": "CardioSense",
    "product_code": "DXH",
    "device_class": "II",
    "udi_di": "00820000077777",
    "recall_number_link": "Z-0456-2023",
    "narrative": "User reported missing contraindication statement in printed IFU within shipped kit."
  },
  {
    "adverse_event_id": "ADR-2024-000301",
    "report_date": "2024-02-02",
    "event_type": "Malfunction",
    "patient_outcome": "No injury",
    "device_problem": "Software Problem",
    "manufacturer_name": "OrthoWorks Ltd.",
    "brand_name": "OrthoAlign",
    "product_code": "HDD",
    "device_class": "II",
    "udi_di": "00990000111111",
    "recall_number_link": "Z-0999-2024",
    "narrative": "Navigation display lag observed; device restarted; procedure continued with alternative guidance."
  }
]
```

### 11.4 Mock GUDID Dataset
```json
[
  {
    "primary_di": "00810000012345",
    "udi_di": "00810000012345",
    "device_description": "NovaPulse Infusion Pump, wireless-enabled, battery-powered.",
    "device_class": "II",
    "manufacturer_name": "NovaMed Systems, Inc.",
    "brand_name": "NovaPulse",
    "product_code": "FRN",
    "gmdn_term": "Infusion pump",
    "mri_safety": "MR Unsafe",
    "sterile": false,
    "single_use": false,
    "implantable": false,
    "contains_nrl": false,
    "version_or_model_number": "NP-2",
    "catalog_number": "NP2-BASE",
    "record_status": "Published",
    "publish_date": "2024-05-30",
    "company_contact_email": "support@novamed.example",
    "company_contact_phone": "+1-555-0100",
    "company_state": "CA",
    "company_country": "US"
  },
  {
    "primary_di": "00820000077777",
    "udi_di": "00820000077777",
    "device_description": "CardioSense ECG Patch, single-use wearable sensor.",
    "device_class": "II",
    "manufacturer_name": "HeartArc Medical",
    "brand_name": "CardioSense",
    "product_code": "DXH",
    "gmdn_term": "Electrocardiograph monitor",
    "mri_safety": "MR Conditional",
    "sterile": true,
    "single_use": true,
    "implantable": false,
    "contains_nrl": false,
    "version_or_model_number": "CS-1",
    "catalog_number": "CS-ECG-PATCH",
    "record_status": "Published",
    "publish_date": "2023-09-01",
    "company_contact_email": "qa@heartarc.example",
    "company_contact_phone": "+1-555-0200",
    "company_state": "MA",
    "company_country": "US"
  },
  {
    "primary_di": "00990000111111",
    "udi_di": "00990000111111",
    "device_description": "OrthoAlign Surgical Navigation System (software and workstation).",
    "device_class": "II",
    "manufacturer_name": "OrthoWorks Ltd.",
    "brand_name": "OrthoAlign",
    "product_code": "HDD",
    "gmdn_term": "Surgical navigation system",
    "mri_safety": "Not evaluated",
    "sterile": false,
    "single_use": false,
    "implantable": false,
    "contains_nrl": false,
    "version_or_model_number": "OA-3",
    "catalog_number": "OA3-WS",
    "record_status": "Published",
    "publish_date": "2024-01-05",
    "company_contact_email": "security@orthoworks.example",
    "company_contact_phone": "+44-20-0000-0000",
    "company_state": "",
    "company_country": "UK"
  }
]
```

**Expected cross-links for validation:**
- Query “NovaPulse” should connect:
  - 510(k) `K240123`
  - Recall `Z-1234-2024`
  - ADR events with `recall_number_link=Z-1234-2024` and `udi_di=00810000012345`
  - GUDID record `udi_di=00810000012345`
- These shared entities must be highlighted in coral and appear in the relationship explorer.

---

## 12) Document Intake + OCR + Markdown Reorganization (Keep + Integrate with Search)

The system must keep the existing doc workflow and integrate its output into Search Studio as optional context.

### 12.1 Input Modes
User can:
- Paste **text/markdown**
- Upload files: **txt, md, pdf**
- Preview PDFs in an embedded **PDF viewer**

### 12.2 PDF Trim + OCR Requirements
- User selects page ranges to **trim**
- User chooses OCR approach:
  1. PyPDF2 text extraction
  2. Local OCR (Tesseract)
  3. LLM-based vision OCR (OpenAI/Gemini)

**User control:**
- Choose which pages to OCR
- Choose OCR engine
- Download trimmed PDF

### 12.3 Markdown Reorganization
After extraction/OCR:
- System reorganizes content into Markdown
- Highlights keywords in **coral**
- User edits in markdown/text view

### 12.4 Using Document Context for Search & Suggestions
Search Studio can optionally offer:
- “Use document terms to propose search queries”
- “Extract candidate device names / product codes / UDIs from doc”
This is an assistive feature; user remains in control.

---

## 13) Managing `agents.yaml` and `SKILL.md` (Upload/Download/Standardize)

### 13.1 Required UI Capabilities
Provide a management feature where user can:
- Download current `agents.yaml` and `SKILL.md`
- Upload replacements for either file
- Edit both in-app (text editor)
- Validate structure

### 13.2 agents.yaml Standardization
If uploaded `agents.yaml` is not standardized:
- System transforms it into standardized schema:
  - `version`
  - `agents[]` with keys:
    - `id`, `name`, `description`
    - `provider`, `model`
    - `temperature`, `max_tokens`
    - `system_prompt`, `user_prompt`

**Standardization outputs:**
- A “Standardization report” summarizing:
  - which fields mapped from where
  - which were missing and defaulted
- A “Diff” view between original and standardized YAML (unified diff)

**Optional LLM-based standardization:**
- If enabled, user selects provider/model and runs an agent-like conversion:
  - Must output YAML only
  - Must preserve content conservatively

### 13.3 SKILL.md Editing
- SKILL.md acts as a global constraint appended to system prompts.
- Must support upload/download/edit and be used in agent runs on:
  - documents
  - selected search results
  - pinned records

---

## 14) Acceptance Criteria (Functional + UX)

### 14.1 Core Search Studio
- A single query returns separated results for all four datasets (when toggled on).
- Each dataset panel shows:
  - hit count
  - charts
  - table
  - row click opens detail drawer

### 14.2 Relationship Highlighting
- Shared entities appear highlighted in coral:
  - at minimum: manufacturer_name, product_code, recall_number, udi_di
- Relationship Explorer identifies cross-dataset bridges and supports click-to-filter.

### 14.3 Suggestions
- After search, system provides at least 5 suggestions:
  - entity-based refinements
  - cross-dataset pivots (e.g., from recall number to ADR)
- Suggestions are clickable and update query/filters.

### 14.4 Agent Integration
- User can run an agent on:
  - a single record
  - filtered dataset results
  - linked cross-dataset bundle
- User can edit prompt and choose model before running.
- Outputs are editable and can be pipelined.

### 14.5 API Key UX
- If env key exists: no input shown; status indicates environment-authenticated.
- If env key missing: show password field; store session-only.

---

## 15) Non-Functional Requirements

- **Performance:** Must remain responsive on mock datasets. For large uploaded datasets:
  - provide result limits
  - avoid rendering huge tables by default
  - progressive disclosure (charts first, table paginated/limited)
- **Explainability:** Show why a record matched (top matched fields).
- **Auditability:** Agent runs stored with timestamp, model, prompts, and input snapshot.
- **Safety:** No fabrication; highlight “Gap” where data missing.
