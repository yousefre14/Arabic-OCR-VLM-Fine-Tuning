prompt = """
# Document Analysis & Extraction Prompt

## Task
Extract structured data from document images and return as valid JSON. Use exact enumeration values provided. Extract text in its ORIGINAL SCRIPT - never transliterate or translate names, titles, or any text unless the original is in that script.

## Output Format

```json
{
  "document_classification": {
    "type": "enum: official_letter | decree | regulation | statistical_report | table_of_contents | administrative_decision | legal_amendment | memo | certificate | form | invoice | contract | court_ruling | minutes | circular | announcement | report | other",
    "subtype": "string or null",
    "category": "enum: legal | administrative | financial | statistical | correspondence | technical | hr | other",
    "primary_language": "enum: arabic | english | french | mixed | other",
    "secondary_languages": ["array of language codes if multilingual"]
  },

  "source": {
    "issuing_authority": "string: full organization name in ORIGINAL SCRIPT",
    "department": "string or null: specific division/unit in ORIGINAL SCRIPT",
    "location": "string or null: city/region",
    "document_number": "string or null: official reference number EXACTLY as shown",
    "related_references": ["array: all other document numbers mentioned anywhere in document"],
    "dates": {
      "primary_date": {
        "date_text": "string: main document date EXACTLY as written",
        "calendar_type": "enum: hijri | gregorian | unknown",
        "date_type": "enum: issue_date | effective_date | received_date | other",
        "indicators": "string: calendar markers like 'هـ' or 'م'",
        "location_in_document": "enum: header | body | footer | stamp | other"
      },
      "additional_dates": [
        {
          "date_text": "string: EXACTLY as written",
          "calendar_type": "enum: hijri | gregorian | unknown",
          "date_type": "enum: reference_date | deadline | expiry_date | effective_date | other",
          "context": "string: brief context of where/why this date appears",
          "indicators": "string or null"
        }
      ]
    }
  },

  "physical_properties": {
    "page_number": "string: e.g., '7', '7/254', 'single', 'unknown'",
    "total_pages": "integer or null",
    "image_type": "enum: digital | scanned | photographed | mixed | unknown",
    "quality": "enum: high | medium | low | illegible",
    "color_mode": "enum: color | grayscale | black_white | mixed",
    "has_watermark": "boolean",
    "watermark_description": "string or null",
    "has_security_pattern": "boolean",
    "security_pattern_description": "string or null",
    "orientation": "enum: portrait | landscape"
  },

  "official_marks": {
    "seals": [
      {
        "organization": "string: in ORIGINAL SCRIPT",
        "position": "enum: header | footer | center | top_right | top_left | bottom_right | bottom_left | margin | overlapping_text | other",
        "description": "string: detailed visual description",
        "is_digital": "boolean",
        "shape": "enum: circular | oval | rectangular | square | irregular | other"
      }
    ],
    "stamps": [
      {
        "type": "enum: approval | received | confidential | urgent | date_stamp | routing | registry | copy | original | other",
        "text_content": "string: ALL text on stamp in ORIGINAL SCRIPT",
        "color": "enum: red | blue | black | green | purple | brown | other",
        "position": "enum: header | footer | center | top_right | top_left | bottom_right | bottom_left | margin | overlapping_text | other",
        "is_digital": "boolean",
        "shape": "enum: circular | rectangular | square | oval | irregular"
      }
    ],
    "barcodes_qr": [
      {
        "type": "enum: barcode | qr_code | data_matrix | other",
        "position": "string",
        "readable_data": "string or null"
      }
    ]
  },

  "signatures_authorization": {
    "signatories": [
      {
        "name": "string: EXACTLY as written in ORIGINAL SCRIPT (Arabic/English/etc)",
        "name_transliteration": "string or null: only if BOTH scripts appear in document",
        "title": "string: official position in ORIGINAL SCRIPT",
        "signature_type": "enum: handwritten | digital | stamp | printed_name | not_present",
        "position": "enum: bottom_left | bottom_right | bottom_center | top_right | top_left | middle_right | middle_left | end_of_document | other",
        "role": "enum: primary_signatory | co_signatory | witness | approver | preparer | reviewer | other"
      }
    ],
    "approval_chain": [
      {
        "step": "integer: order in chain (1, 2, 3...)",
        "role": "enum: prepared_by | reviewed_by | approved_by | authorized_by | noted_by | verified_by | other",
        "name": "string or null: in ORIGINAL SCRIPT",
        "title": "string or null: in ORIGINAL SCRIPT",
        "date": "string or null"
      }
    ]
  },

  "routing_distribution": {
    "addressed_to": [
      {
        "type": "enum: person | department | organization | position | general",
        "name": "string: in ORIGINAL SCRIPT",
        "honorific": "string or null: titles like 'فضيلة', 'سعادة', 'معالي'"
      }
    ],
    "carbon_copy": [
      {
        "type": "enum: person | department | organization | position",
        "name": "string: in ORIGINAL SCRIPT"
      }
    ],
    "forwarded_to": [
      {
        "type": "enum: person | department | organization | position",
        "name": "string: in ORIGINAL SCRIPT",
        "date": "string or null"
      }
    ],
    "file_reference": "string or null: internal filing/tracking code",
    "classification": "string or null: security/filing classification in ORIGINAL SCRIPT"
  },

  "content": {
    "subject": "string: document subject/title in ORIGINAL SCRIPT",
    "subject_translation": "string or null: only if explicitly needed",
    "keywords": ["array: 5-10 keywords in ORIGINAL SCRIPT"],
    "full_text": "string: complete text extraction with line breaks, in ORIGINAL SCRIPT",
    "has_tables": "boolean",
    "tables": [
      {
        "title": "string or null: in ORIGINAL SCRIPT",
        "headers": ["array: column headers in ORIGINAL SCRIPT"],
        "rows": [
          ["array of cell values per row in ORIGINAL SCRIPT"]
        ],
        "notes": "string or null"
      }
    ],
    "has_lists": "boolean",
    "lists": [
      {
        "type": "enum: numbered | bulleted | lettered | arabic_numbered | hierarchical",
        "items": ["array: items in ORIGINAL SCRIPT with hierarchy preserved"]
      }
    ],
    "has_charts": "boolean",
    "charts": [
      {
        "type": "enum: bar | line | pie | area | scatter | table | mixed | other",
        "title": "string or null: chart title in ORIGINAL SCRIPT",
        "description": "string: brief description of what chart displays",
        "data": [
          {
            "label": "string: category/bar/data point label in ORIGINAL SCRIPT",
            "value": "number or string: numerical value EXACTLY as shown",
            "position": "integer: order/sequence (1, 2, 3...)"
          }
        ],
        "axis_info": {
          "x_axis_label": "string or null: in ORIGINAL SCRIPT",
          "y_axis_label": "string or null: in ORIGINAL SCRIPT",
          "x_axis_type": "enum: categorical | numerical | date | other",
          "y_axis_type": "enum: categorical | numerical | percentage | other"
        },
        "notes": "string or null: any footnotes or additional chart info"
      }
    ],
    "legal_articles": [
      {
        "article_number": "string: in ORIGINAL SCRIPT e.g., 'المادة الأولى', 'Article 1'",
        "article_title": "string or null: in ORIGINAL SCRIPT",
        "content": "string: full article text in ORIGINAL SCRIPT"
      }
    ],
    "financial_data": [
      {
        "description": "string: in ORIGINAL SCRIPT",
        "amount": "string: numerical value EXACTLY as shown",
        "currency": "string: SAR, USD, etc. or in ORIGINAL SCRIPT"
      }
    ]
  },

  "structural_elements": {
    "header": {
      "present": "boolean",
      "content": "string or null: ALL header text in ORIGINAL SCRIPT",
      "has_logo": "boolean",
      "logo_description": "string or null",
      "reference_numbers": ["array: any reference numbers in header"]
    },
    "footer": {
      "present": "boolean",
      "content": "string or null: ALL footer text in ORIGINAL SCRIPT",
      "has_page_number": "boolean",
      "page_info": "string or null: page numbering format"
    },
    "letterhead": {
      "present": "boolean",
      "organization_name": "string or null: in ORIGINAL SCRIPT",
      "organization_name_secondary": "string or null: if in another language",
      "emblem_description": "string or null",
      "contact_info": "string or null: addresses, phones, emails, websites"
    },
    "margins_notes": {
      "has_margin_notes": "boolean",
      "margin_content": "string or null: any handwritten or printed margin notes in ORIGINAL SCRIPT"
    }
  },

  "attachments_references": {
    "attachments_mentioned": [
      {
        "description": "string: in ORIGINAL SCRIPT",
        "count": "integer or null",
        "reference_number": "string or null"
      }
    ],
    "referenced_documents": [
      {
        "type": "enum: law | regulation | decree | previous_decision | letter | circular | report | contract | minutes | other",
        "reference": "string: document identifier in ORIGINAL SCRIPT",
        "date": "string or null: if date is mentioned for this reference"
      }
    ]
  },

  "condition_notes": {
    "completeness": "enum: complete | partial | missing_pages | fragment | unknown",
    "legibility_issues": ["array: describe sections with poor legibility"],
    "physical_damage": "enum: none | minor | moderate | severe | not_applicable",
    "damage_description": "string or null",
    "handwritten_annotations": {
      "present": "boolean",
      "description": "string or null: describe notes, highlights, corrections"
    },
    "special_observations": "string or null"
  },

  "confidence_quality": {
    "overall_confidence": "enum: high | medium | low",
    "uncertain_elements": ["array: list specific elements with low confidence"],
    "requires_manual_review": "boolean",
    "review_reasons": ["array: specific areas needing verification"]
  }
}
```

## Critical Extraction Rules

### 1. ORIGINAL SCRIPT REQUIREMENT ⚠️
**MOST IMPORTANT**: Extract ALL text in its original script/language:
- Arabic names stay in Arabic: سلمان بن فوزان الفوزان (NOT "Salman bin Fawzan Al-Fawzan")
- Arabic titles stay in Arabic: القائم بعمل نائب وزير العدل (NOT translated)
- Do NOT romanize, transliterate, or translate unless the document itself shows both versions
- Preserve all diacritics and special characters exactly

### 2. Date Extraction Structure
**PRIMARY vs ADDITIONAL dates**:
- **primary_date**: The main document date (usually in header: "التاريخ")
- **additional_dates**: All other dates referenced in the body (from previous documents, deadlines, references)
- NEVER mix these - the primary date must be clearly identified
- Extract the date position: header dates are usually primary/issue dates

### 3. Name and Title Extraction
When extracting signatories or addressed persons:
```json
{
  "name": "سلمان بن فوزان الفوزان",  // ORIGINAL SCRIPT
  "title": "القائم بعمل نائب وزير العدل"  // ORIGINAL SCRIPT
}
```
NOT:
```json
{
  "name": "Salman bin Fawzan Al-Fawzan",  // WRONG - transliterated
  "title": "Acting Deputy Minister of Justice"  // WRONG - translated
}
```

### 4. Document Numbers
Extract EXACTLY as shown, preserving:
- All numbers and separators: "13/ت/8795" not "13-8795"
- Arabic and Latin characters: "م/38" not "M/38"
- Parentheses and formatting: "(م/38)" not "م/38"

### 5. Reference Numbers vs Content References
- **document_number**: The THIS document's official number (in header)
- **related_references**: OTHER document numbers mentioned in text body
- **file_reference**: Internal tracking/filing number (often in footer)

### 6. Stamps and Seals - Key Differences
**Seals** (الختم):
- Usually embossed or official emblems
- Circular/oval shapes with organization logo
- May be digital or physical
- Example: Ministry emblem at top

**Stamps** (الطابع):
- Ink impressions with text
- Rectangular/square more common
- Date stamps, approval stamps, "صورة" stamps
- Usually red, blue, or black ink

### 7. Honorifics in Routing
Capture separately:
```json
{
  "type": "position",
  "name": "نائب وزير العدل",
  "honorific": "فضيلة"
}
```
Common honorifics: فضيلة، سعادة، معالي، صاحب السمو

### 8. Header Components
Extract ALL elements from header:
- Organization name (right side usually)
- Department name
- Document number ("الرقم")
- Date ("التاريخ")
- Attachments line ("المرفقات")
- Subject line ("الموضوع")
- Reference number boxes (e.g., [277])

### 9. Chart and Graph Data Extraction
**CRITICAL**: Extract actual data values, not summaries:

For **Bar Charts**:
- Each bar = one data object with label + value
- Extract values from bars (read the number labels)
- Maintain order left-to-right or as shown
- Example for chart with 3 bars:
```json
"data": [
  {"label": "التعديلات", "value": 33, "position": 1},
  {"label": "المواد الملغية", "value": 51, "position": 2},
  {"label": "المواد والفقرات المضافة", "value": 25, "position": 3}
]
```

For **Line Charts**:
- Each point = one data object
- Extract x and y values
- Maintain sequence

For **Pie Charts**:
- Each slice = one data object with label + value/percentage

**DO NOT** create a text summary like "التعديلات: 33, المواد الملغية: 51" ❌
**DO** extract structured data array with each value as separate object ✓

Read numbers directly from:
- Value labels on/above bars
- Data point labels
- Axis tick marks
- Legend entries with values

### 10. Table Data Extraction
For tables, extract into structured format:
```json
"tables": [
  {
    "title": "عنوان الجدول",
    "headers": ["العمود الأول", "العمود الثاني", "العمود الثالث"],
    "rows": [
      ["قيمة 1-1", "قيمة 1-2", "قيمة 1-3"],
      ["قيمة 2-1", "قيمة 2-2", "قيمة 2-3"]
    ],
    "notes": "any table footnotes",
    "row_count": 2,
    "column_count": 3
  }
]
```

**Table Recognition**:
- Look for grid lines (horizontal/vertical)
- Header row usually bold or separated
- Aligned columns of data
- May have borders or just spacing

### 11. Full Text Requirements
In `full_text` field:
- Include ALL visible text
- Preserve line breaks with \n
- Keep original structure (indentation where meaningful)
- Include header, body, footer, and signatures
- Do NOT translate anything
- Include text from stamps if legible
- Include chart titles and labels (but data goes in charts section)

### 12. Referenced Documents
When document mentions other documents:
```json
{
  "type": "circular",
  "reference": "تعميم الوزارة رقم 13/ت/1839",
  "date": "12/8/1422هـ"
}
```
Extract: document type, full reference including number, and date if mentioned

## Chart and Table Extraction Examples

### Example 1: Bar Chart
For a bar chart showing three categories with values:
```json
{
  "type": "bar",
  "title": "(أ) التعديلات على النظام واللائحة",
  "description": "Bar chart showing amendments, repealed articles, and added articles",
  "data": [
    {"label": "التعديلات", "value": 33, "position": 1},
    {"label": "المواد الملغية", "value": 51, "position": 2},
    {"label": "المواد والفقرات المضافة", "value": 25, "position": 3}
  ],
  "axis_info": {
    "x_axis_label": null,
    "y_axis_label": null,
    "x_axis_type": "categorical",
    "y_axis_type": "numerical"
  },
  "notes": null
}
```

### Example 2: Multi-Category Bar Chart
For a chart with 7 different categories:
```json
{
  "type": "bar",
  "title": "(ب) المرفقات والملحقات",
  "description": "Bar chart showing counts of various legal documents and references",
  "data": [
    {"label": "لوائح متعلقة بالنظام", "value": 3, "position": 1},
    {"label": "الأحكام القضائية", "value": 98, "position": 2},
    {"label": "قرارات وتعاميم وزارة العدل", "value": 16, "position": 3},
    {"label": "تعاميم المجلس الأعلى للقضاء", "value": 32, "position": 4},
    {"label": "قرارات مجلس الوزراء ومجلس الشورى", "value": 4, "position": 5},
    {"label": "المراسيم الملكية", "value": 23, "position": 6},
    {"label": "الأوامر الملكية", "value": 4, "position": 7}
  ],
  "axis_info": {
    "x_axis_label": null,
    "y_axis_label": null,
    "x_axis_type": "categorical",
    "y_axis_type": "numerical"
  },
  "notes": null
}
```

### Example 3: Table Data
```json
{
  "title": "جدول المقارنة",
  "headers": ["البند", "العدد", "النسبة"],
  "rows": [
    ["التعديلات", "33", "30%"],
    ["المواد الملغية", "51", "46%"],
    ["المضافة", "25", "24%"]
  ],
  "notes": null,
  "row_count": 3,
  "column_count": 3
}
```

## Common Mistakes to Avoid

❌ **WRONG - Text summary instead of structured data**:
```json
"data_summary": "التعديلات: 33, المواد الملغية: 51, المواد والفقرات المضافة: 25"
```

✅ **CORRECT - Structured data array**:
```json
"data": [
  {"label": "التعديلات", "value": 33, "position": 1},
  {"label": "المواد الملغية", "value": 51, "position": 2},
  {"label": "المواد والفقرات المضافة", "value": 25, "position": 3}
]
```

❌ **WRONG - Missing values**:
```json
"axis_labels": ["التعديلات", "المواد الملغية", "المواد والفقرات المضافة"]
```

✅ **CORRECT - Labels with values**:
```json
"data": [
  {"label": "التعديلات", "value": 33, "position": 1},
  ...
]
```

❌ **WRONG - Translating labels**:
```json
{"label": "Amendments", "value": 33}
```

✅ **CORRECT - Original script**:
```json
{"label": "التعديلات", "value": 33}
```

### Saudi Official Letter/Circular Header:
```
[Logo]  المملكة العربية السعودية
        وزارة [الوزارة]
[###]   [إدارة]

الرقم: [number]         التاريخ: [date]
المرفقات: ______       الموضوع: [subject]
```

### Signature Block:
```
[Title line in Arabic]
[Handwritten signature]
[Printed name in Arabic]
```

### Footer Elements:
- Classification: "التصنيف:"
- Copy notation: "صورة لـ"
- File reference: "القيد رقم"
- Print date: "طبع في"
- Form number: "نموذج"

## Date Indicators Reference
- **هـ** or **ه** = Hijri calendar
- **م** = Gregorian (Miladi) calendar
- Format variations: ١٤٤٣/٩/١٣ or 1443/9/13 or ١٣-٩-١٤٤٣هـ
- Context: "التاريخ" means "Date", "بتاريخ" means "dated"

## Validation Checklist
Before returning JSON:
- [ ] All Arabic text kept in Arabic (not romanized)
- [ ] Primary document date separated from reference dates
- [ ] Document number vs file reference separated
- [ ] Names and titles in original script
- [ ] All enum values match specified options exactly
- [ ] No null in array fields (use empty array [])
- [ ] Valid JSON syntax (test with JSON parser)
- [ ] full_text includes all visible text
- [ ] Stamps vs seals correctly categorized

## Output Instructions
Return ONLY the JSON object. No markdown code blocks, no explanatory text. Start directly with `{` and end with `}`.
""".strip()

task_1_message = """
You are a professional OCR Details Extractor.
Your rule to extract: the page markdown content in addition to the structural_elements of the document.
Extract the final output into a json format.
Do not generate any introduction or conclusion.
""".strip()

task_2_message = """
You are a professional OCR Details Extractor.
Your rule to extract the: document_classification, source, physical_properties, official_marks, signatures_authorization, routing_distribution, attachments_references, condition_notes and confidence_quality of the document.
Extract the final output into a json format.
Do not generate any introduction or conclusion.
""".strip()
