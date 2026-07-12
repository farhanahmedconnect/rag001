#!/usr/bin/env python
"""
Generate a comprehensive 10-page PDF about Insurance Health Claim Rules
using PyMuPDF (fitz). Includes embedded images, headers, footers, and
rich domain-accurate content.
"""

import fitz  # PyMuPDF
import os
import sys

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_PDF = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "document.pdf")
IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")

IMAGES = {
    "flowchart": os.path.join(IMG_DIR, "claim_process_flowchart_1783874513069.png"),
    "denial":    os.path.join(IMG_DIR, "claim_denial_reasons_1783874523502.png"),
    "eligibility": os.path.join(IMG_DIR, "eligibility_verification_1783874535782.png"),
    "coverage":  os.path.join(IMG_DIR, "coverage_tiers_1783874557613.png"),
    "appeals":   os.path.join(IMG_DIR, "appeals_process_1783874568110.png"),
    "mainframe": os.path.join(IMG_DIR, "mainframe_screen_1783875258821.png"),
    "adjudication": os.path.join(IMG_DIR, "mainframe_adjudication_1783875277552.png"),
}

# Page dimensions (Letter size in points)
PAGE_W, PAGE_H = fitz.paper_size("letter")
MARGIN_LEFT = 54
MARGIN_RIGHT = 54
MARGIN_TOP = 72
MARGIN_BOTTOM = 54
CONTENT_W = PAGE_W - MARGIN_LEFT - MARGIN_RIGHT
HEADER_Y = 36
FOOTER_Y = PAGE_H - 36

# Font sizes
TITLE_SIZE = 26
SUBTITLE_SIZE = 16
HEADING_SIZE = 14
SUBHEADING_SIZE = 11.5
BODY_SIZE = 9.5
SMALL_SIZE = 8
TABLE_SIZE = 8.5

# Colors
COLOR_DARK_BLUE = (0.05, 0.15, 0.35)
COLOR_BLUE = (0.1, 0.25, 0.55)
COLOR_BLACK = (0, 0, 0)
COLOR_GRAY = (0.35, 0.35, 0.35)
COLOR_LIGHT_GRAY = (0.55, 0.55, 0.55)
COLOR_WHITE = (1, 1, 1)
COLOR_ACCENT = (0.0, 0.42, 0.64)
COLOR_HEADER_BG = (0.05, 0.15, 0.35)
COLOR_TABLE_HEADER = (0.08, 0.20, 0.45)
COLOR_TABLE_ALT = (0.93, 0.95, 0.98)

FONT_BODY = "helv"        # Helvetica
FONT_BOLD = "hebo"        # Helvetica-Bold
FONT_ITALIC = "heit"      # Helvetica-Oblique


def add_header_footer(page, page_num, total_pages, show_header=True):
    """Add header bar and footer to a page."""
    if show_header:
        # Header background bar
        header_rect = fitz.Rect(0, 0, PAGE_W, 52)
        page.draw_rect(header_rect, color=None, fill=COLOR_HEADER_BG)
        # Header text
        page.insert_text(
            fitz.Point(MARGIN_LEFT, 20),
            "HealthGuard Insurance Corporation",
            fontname=FONT_BOLD, fontsize=9, color=COLOR_WHITE
        )
        page.insert_text(
            fitz.Point(MARGIN_LEFT, 35),
            "Health Claim Rules & Guidelines  |  Version 3.2  |  Effective: January 1, 2024",
            fontname=FONT_BODY, fontsize=7, color=(0.7, 0.78, 0.9)
        )
        # Divider line under header
        page.draw_line(
            fitz.Point(MARGIN_LEFT, 54),
            fitz.Point(PAGE_W - MARGIN_RIGHT, 54),
            color=COLOR_ACCENT, width=0.5
        )

    # Footer line
    page.draw_line(
        fitz.Point(MARGIN_LEFT, PAGE_H - 48),
        fitz.Point(PAGE_W - MARGIN_RIGHT, PAGE_H - 48),
        color=COLOR_LIGHT_GRAY, width=0.4
    )
    # Footer text
    page.insert_text(
        fitz.Point(MARGIN_LEFT, PAGE_H - 34),
        "CONFIDENTIAL - HealthGuard Insurance Corporation - Internal Use & Provider Reference",
        fontname=FONT_ITALIC, fontsize=6.5, color=COLOR_LIGHT_GRAY
    )
    page.insert_text(
        fitz.Point(PAGE_W - MARGIN_RIGHT - 60, PAGE_H - 34),
        f"Page {page_num} of {total_pages}",
        fontname=FONT_BODY, fontsize=7, color=COLOR_GRAY
    )


def insert_wrapped_text(page, text, start_point, fontname, fontsize, color,
                         max_width, line_height=None):
    """Insert word-wrapped text and return the Y position after the last line."""
    if line_height is None:
        line_height = fontsize * 1.45
    x, y = start_point.x, start_point.y
    words = text.split()
    line = ""
    for word in words:
        test_line = f"{line} {word}".strip()
        tw = fitz.get_text_length(test_line, fontname=fontname, fontsize=fontsize)
        if tw > max_width and line:
            page.insert_text(fitz.Point(x, y), line,
                             fontname=fontname, fontsize=fontsize, color=color)
            y += line_height
            line = word
        else:
            line = test_line
    if line:
        page.insert_text(fitz.Point(x, y), line,
                         fontname=fontname, fontsize=fontsize, color=color)
        y += line_height
    return y


def insert_heading(page, text, y, level=1):
    """Insert a section heading and return new Y."""
    if level == 1:
        fs, fn, clr = HEADING_SIZE, FONT_BOLD, COLOR_DARK_BLUE
    elif level == 2:
        fs, fn, clr = SUBHEADING_SIZE, FONT_BOLD, COLOR_BLUE
    else:
        fs, fn, clr = BODY_SIZE + 1, FONT_BOLD, COLOR_BLACK
    y_after = insert_wrapped_text(page, text, fitz.Point(MARGIN_LEFT, y),
                                   fn, fs, clr, CONTENT_W)
    # underline for level 1
    if level == 1:
        page.draw_line(fitz.Point(MARGIN_LEFT, y + 3),
                       fitz.Point(MARGIN_LEFT + CONTENT_W, y + 3),
                       color=COLOR_ACCENT, width=0.7)
    return y_after + 2


def insert_body(page, text, y, indent=0):
    """Insert body paragraph and return new Y."""
    return insert_wrapped_text(
        page, text, fitz.Point(MARGIN_LEFT + indent, y),
        FONT_BODY, BODY_SIZE, COLOR_BLACK, CONTENT_W - indent
    )


def insert_body_bold(page, text, y, indent=0):
    """Insert bold body text."""
    return insert_wrapped_text(
        page, text, fitz.Point(MARGIN_LEFT + indent, y),
        FONT_BOLD, BODY_SIZE, COLOR_BLACK, CONTENT_W - indent
    )


def insert_image(page, img_path, y, img_width=None, center=True):
    """Insert an image centered on the page. Return Y after image."""
    if not os.path.exists(img_path):
        y = insert_body(page, f"[Image not found: {os.path.basename(img_path)}]", y)
        return y + 10
    img = fitz.open(img_path)
    img_page = img[0]
    iw, ih = img_page.rect.width, img_page.rect.height
    img.close()
    if img_width is None:
        img_width = CONTENT_W * 0.48
    scale = img_width / iw
    img_height = ih * scale
    # Check if image fits on page
    if y + img_height > PAGE_H - MARGIN_BOTTOM - 30:
        img_height = PAGE_H - MARGIN_BOTTOM - 30 - y
        scale = img_height / ih
        img_width = iw * scale
    if center:
        x = MARGIN_LEFT + (CONTENT_W - img_width) / 2
    else:
        x = MARGIN_LEFT
    rect = fitz.Rect(x, y, x + img_width, y + img_height)
    page.insert_image(rect, filename=img_path)
    return y + img_height + 8


def draw_table(page, headers, rows, y, col_widths=None):
    """Draw a simple table. Return Y after table."""
    num_cols = len(headers)
    if col_widths is None:
        col_widths = [CONTENT_W / num_cols] * num_cols
    row_height = 14
    x_start = MARGIN_LEFT

    # Header row
    x = x_start
    for i, hdr in enumerate(headers):
        rect = fitz.Rect(x, y, x + col_widths[i], y + row_height)
        page.draw_rect(rect, color=None, fill=COLOR_TABLE_HEADER)
        page.insert_text(fitz.Point(x + 3, y + 10), hdr,
                         fontname=FONT_BOLD, fontsize=TABLE_SIZE, color=COLOR_WHITE)
        x += col_widths[i]
    y += row_height

    # Data rows
    for r_idx, row in enumerate(rows):
        x = x_start
        fill = COLOR_TABLE_ALT if r_idx % 2 == 0 else COLOR_WHITE
        for i, cell in enumerate(row):
            rect = fitz.Rect(x, y, x + col_widths[i], y + row_height)
            page.draw_rect(rect, color=(0.8, 0.8, 0.8), fill=fill, width=0.3)
            page.insert_text(fitz.Point(x + 3, y + 10), str(cell),
                             fontname=FONT_BODY, fontsize=TABLE_SIZE, color=COLOR_BLACK)
            x += col_widths[i]
        y += row_height
    return y + 6


# ============================================================================
# PAGE CONTENT FUNCTIONS
# ============================================================================

def page1_title(doc):
    """Page 1 - Title Page"""
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    # Full-page background gradient effect (dark blue block at top)
    top_block = fitz.Rect(0, 0, PAGE_W, PAGE_H * 0.45)
    page.draw_rect(top_block, color=None, fill=COLOR_DARK_BLUE)

    # Decorative accent line
    page.draw_line(fitz.Point(PAGE_W * 0.15, PAGE_H * 0.46),
                   fitz.Point(PAGE_W * 0.85, PAGE_H * 0.46),
                   color=COLOR_ACCENT, width=2.5)

    # Title text on dark background
    y = 140
    page.insert_text(fitz.Point(PAGE_W / 2 - 180, y), "INSURANCE",
                     fontname=FONT_BOLD, fontsize=36, color=COLOR_WHITE)
    y += 48
    page.insert_text(fitz.Point(PAGE_W / 2 - 180, y), "HEALTH CLAIM",
                     fontname=FONT_BOLD, fontsize=36, color=COLOR_WHITE)
    y += 48
    page.insert_text(fitz.Point(PAGE_W / 2 - 180, y), "RULES & GUIDELINES",
                     fontname=FONT_BOLD, fontsize=36, color=(0.55, 0.78, 1.0))

    # Subtitle area below blue block
    y = PAGE_H * 0.52
    page.insert_text(fitz.Point(MARGIN_LEFT + 40, y), "Version 3.2",
                     fontname=FONT_BOLD, fontsize=18, color=COLOR_DARK_BLUE)
    y += 32
    page.insert_text(fitz.Point(MARGIN_LEFT + 40, y),
                     "Effective Date: January 1, 2024",
                     fontname=FONT_BODY, fontsize=14, color=COLOR_GRAY)
    y += 28
    page.insert_text(fitz.Point(MARGIN_LEFT + 40, y),
                     "HealthGuard Insurance Corporation",
                     fontname=FONT_BOLD, fontsize=14, color=COLOR_BLUE)
    y += 24
    page.insert_text(fitz.Point(MARGIN_LEFT + 40, y),
                     "Claims Administration Division",
                     fontname=FONT_BODY, fontsize=11, color=COLOR_GRAY)
    y += 50

    # Document info box
    info_rect = fitz.Rect(MARGIN_LEFT + 30, y, PAGE_W - MARGIN_RIGHT - 30, y + 120)
    page.draw_rect(info_rect, color=COLOR_ACCENT, fill=(0.95, 0.97, 1.0), width=0.8)
    y += 18
    info_items = [
        "Document Classification: CONFIDENTIAL - Internal & Provider Use",
        "Approved By: Dr. Sarah Mitchell, VP Claims Operations",
        "Review Cycle: Annual (Next Review: January 1, 2025)",
        "Applicable Jurisdictions: All 50 U.S. States & District of Columbia",
        "Supersedes: Version 3.1 dated July 1, 2023",
        "Distribution: Claims Adjusters, Medical Directors, Provider Relations",
    ]
    for item in info_items:
        page.insert_text(fitz.Point(MARGIN_LEFT + 42, y), item,
                         fontname=FONT_BODY, fontsize=8.5, color=COLOR_BLACK)
        y += 16

    add_header_footer(page, 1, 10, show_header=False)


def page2_overview(doc):
    """Page 2 - Health Insurance Claim Lifecycle Overview"""
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    add_header_footer(page, 2, 10)
    y = MARGIN_TOP + 4

    y = insert_heading(page, "1. Health Insurance Claim Lifecycle Overview", y, level=1)
    y += 4

    y = insert_heading(page, "1.1 What Is a Health Insurance Claim?", y, level=2)
    y = insert_body(page, (
        "A health insurance claim is a formal request submitted by either a healthcare provider or an insured "
        "member to the insurance payer seeking reimbursement or direct payment for medical services rendered. "
        "Each claim contains detailed information about the patient encounter, including diagnosis codes "
        "(ICD-10-CM), procedure codes (CPT/HCPCS), dates of service, provider identifiers (NPI), and billed "
        "amounts. The claim serves as the primary transactional document in the revenue cycle and triggers a "
        "complex adjudication workflow that determines the financial responsibility of each party involved. "
        "HealthGuard Insurance processes approximately 4.2 million claims annually across its network of over "
        "85,000 contracted providers, with an average turnaround time of 12.3 business days from receipt to "
        "payment determination."
    ), y)
    y += 4

    y = insert_heading(page, "1.2 Parties Involved in the Claims Process", y, level=2)
    y = insert_body(page, (
        "The claims ecosystem involves four primary stakeholders. The Insured (Member) is the individual "
        "covered under the health insurance policy who receives medical services and bears ultimate financial "
        "responsibility for any patient cost-sharing amounts. The Healthcare Provider (facility or physician) "
        "delivers medical services and submits claims either electronically via ANSI X12 837 transactions or on "
        "paper using CMS-1500 (professional) or UB-04 (institutional) forms. The Payer is the insurance "
        "company or managed care organization (in this case, HealthGuard Insurance Corporation) that evaluates "
        "the claim against policy terms, fee schedules, and medical necessity criteria to determine payment. "
        "The Third-Party Administrator (TPA) may be engaged by self-funded employer groups to handle "
        "day-to-day claims processing; HealthGuard serves as TPA for 340 self-funded employer plans covering "
        "approximately 1.1 million additional lives."
    ), y)
    y += 4

    y = insert_heading(page, "1.3 Types of Health Insurance Claims", y, level=2)
    y = insert_body(page, (
        "HealthGuard recognizes two primary claim pathways. Cashless Claims (also known as direct settlement "
        "or network claims) occur when the insured member receives treatment at an in-network hospital or "
        "facility. In this model, the provider contacts HealthGuard's pre-authorization desk prior to or at the "
        "time of admission, and upon discharge the hospital bills HealthGuard directly. The member pays only "
        "the applicable co-payment, co-insurance, or deductible amount at the point of service. Cashless "
        "claims account for approximately 68% of all HealthGuard inpatient claims and carry a target "
        "processing time of 4 hours for pre-authorization and 15 business days for final settlement."
    ), y)
    y += 2
    y = insert_body(page, (
        "Reimbursement Claims (also known as indemnity claims) arise when the insured member receives "
        "treatment at a non-network facility or when pre-authorization was not obtained for network "
        "treatment. The member pays the full cost of treatment upfront and subsequently submits a claim to "
        "HealthGuard with original bills, receipts, discharge summary, and a completed Claim Form HG-101. "
        "Reimbursement claims must be filed within 30 calendar days of discharge (15 days for outpatient "
        "services) and are processed within 30 business days of receipt of complete documentation."
    ), y)
    y += 4

    y = insert_heading(page, "1.4 Lifecycle Stages", y, level=2)
    y = insert_body(page, (
        "Every health insurance claim passes through a defined lifecycle comprising seven stages: (1) Encounter "
        "& Service Delivery, where the medical event occurs; (2) Claim Generation, where the provider's "
        "billing department codes the encounter and generates the claim; (3) Claim Submission, where the claim "
        "is transmitted electronically or mailed to the payer; (4) Intake & Validation, where HealthGuard's "
        "front-end systems verify the claim format, member eligibility, and data completeness; (5) Adjudication, "
        "where the claim is evaluated against medical policies, fee schedules, benefit limits, and coordination "
        "of benefits rules; (6) Payment Determination, where the allowed amount, member responsibility, and "
        "provider payment are calculated; and (7) Remittance & Communication, where the Explanation of Benefits "
        "(EOB) is generated for the member and the Electronic Remittance Advice (ERA/835) is transmitted to the "
        "provider. Claims that fail at any stage enter exception queues for manual review by certified claims "
        "examiners within HealthGuard's Claims Operations Center."
    ), y)


def page3_submission(doc):
    """Page 3 - Claim Submission Process"""
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    add_header_footer(page, 3, 10)
    y = MARGIN_TOP + 4

    y = insert_heading(page, "2. Claim Submission Process", y, level=1)
    y += 2

    y = insert_body(page, (
        "The claim submission process is the critical first step in obtaining payment for medical services "
        "rendered. HealthGuard requires all claims to be submitted in compliance with HIPAA transaction "
        "standards (ANSI X12 837P for professional claims and 837I for institutional claims). Paper "
        "submissions are accepted only from providers with fewer than 25 full-time equivalent employees, "
        "as permitted under HIPAA administrative simplification rules. All claims enter through "
        "HealthGuard's Electronic Data Interchange (EDI) gateway, which performs real-time syntax validation "
        "and issues an immediate 999 Functional Acknowledgment followed by a 277CA Claim Acknowledgment "
        "within 24 hours of receipt."
    ), y)
    y += 4

    # Insert flowchart image
    y = insert_image(page, IMAGES["flowchart"], y, img_width=CONTENT_W * 0.45)
    y += 2

    y = insert_heading(page, "2.1 Required Documentation", y, level=2)
    y = insert_body(page, (
        "Every claim submission must include the following core documentation to be considered complete: "
        "(a) Claim Form - CMS-1500 for professional services or UB-04 for institutional services, fully "
        "completed with all mandatory fields including patient demographics, subscriber information, "
        "diagnosis codes (minimum one primary ICD-10-CM code), procedure codes with modifiers, dates of "
        "service, place of service code, rendering provider NPI, and billed charges per line item. "
        "(b) Itemized Hospital Bills showing date-wise breakdown of all charges including room rent, "
        "nursing charges, investigation charges, pharmacy, consumables, surgeon fees, anesthesia, and OT "
        "charges. (c) Discharge Summary with admission date, discharge date, diagnosis at admission, "
        "diagnosis at discharge, procedures performed, treatment given, and condition at discharge. "
        "(d) Investigation Reports including pathology, radiology, and laboratory results that support the "
        "medical necessity of the treatment provided."
    ), y)
    y += 4

    y = insert_heading(page, "2.2 Filing Timelines & System Processing", y, level=2)
    y = insert_body(page, (
        "Cashless claims must have pre-authorization initiated within 48 hours of emergency admission or "
        "7 days prior to planned admission, with final claim submission within 15 business days of "
        "discharge. Reimbursement claims must be filed within 30 calendar days from the date of discharge "
        "for inpatient services or 15 calendar days from the date of service for outpatient procedures. "
        "Late submissions may be accepted up to 180 days with a valid reason code (RC-LATE-01 through "
        "RC-LATE-05) and management approval. Claims filed beyond 365 days from the date of service are "
        "permanently barred under the timely filing provision of the member's benefit contract."
    ), y)
    y += 2
    y = insert_body(page, (
        "Upon receipt, claims are routed to mainframe screen HC01 (New Claim Entry) in HealthGuard's CICS "
        "Claims Processing System. The intake clerk verifies data completeness using checklist transaction "
        "HC01-CHK, and the system auto-assigns a unique CLAIM-ID in the format HC-YYYY-NNNNN (e.g., "
        "HC-2024-00142). Incomplete claims are pended with status code 'P' and a deficiency letter is "
        "generated within 3 business days, giving the provider 45 days to submit the missing information "
        "before the claim is administratively denied with denial code DN001."
    ), y)


def page4_mainframe(doc):
    """Page 4 - Mainframe Claim Processing System"""
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    add_header_footer(page, 4, 10)
    y = MARGIN_TOP + 4

    y = insert_heading(page, "3. Mainframe Claim Processing System - Screen Reference", y, level=1)
    y += 2

    y = insert_body(page, (
        "HealthGuard's core claims processing infrastructure runs on an IBM z/Series mainframe environment "
        "using CICS (Customer Information Control System) for online transaction processing and JES2 for "
        "batch job scheduling. The system processes an average of 18,000 claims per day during peak periods "
        "and maintains a 99.97% uptime SLA. Below is the primary claim entry and inquiry screen used by "
        "claims examiners."
    ), y)
    y += 4

    # Insert mainframe screen image
    y = insert_image(page, IMAGES["mainframe"], y, img_width=CONTENT_W * 0.50)
    y += 2

    y = insert_heading(page, "3.1 Field Definitions & Validation Rules", y, level=2)
    y = insert_body(page, (
        "CLAIM-ID: System-generated unique identifier in the format HC-YYYY-NNNNN where YYYY represents "
        "the claim year and NNNNN is a zero-padded sequential number resetting annually. The CLAIM-ID is "
        "the primary key in the DB2 CLAIMS_MASTER table and is referenced across all downstream processing. "
        "MEMBER-ID: The subscriber's unique identifier, validated in real-time against the MEMBER_ELIG "
        "VSAM file. The lookup procedure verifies the member's enrollment status, effective dates, group "
        "number, plan code, and benefit tier. An invalid or terminated MEMBER-ID triggers immediate claim "
        "rejection with status code 'R' and denial reason DN-ELIG-001."
    ), y)
    y += 2

    y = insert_body(page, (
        "ICD-10-CM and CPT Code Validation: The system cross-references submitted diagnosis and procedure "
        "code combinations against the HCPCS_VALID_PAIRS DB2 table, which contains over 2.3 million "
        "approved ICD-10/CPT pairings. Invalid combinations trigger edit code E-0042 (Invalid Dx/Px Pair) "
        "and route the claim to the clinical review queue. Procedure codes are also validated against the "
        "provider's taxonomy code to ensure the rendering provider is qualified to perform the reported "
        "service. Modifier codes (25, 26, 59, 76, 77) are validated for appropriate usage according to "
        "CMS National Correct Coding Initiative (NCCI) edits."
    ), y)
    y += 2

    y = insert_heading(page, "3.2 Amount Calculation & CICS Transactions", y, level=2)
    y = insert_body(page, (
        "The Allowed Amount is calculated as: Allowed Amount = min(Billed Amount, UCR Rate), where the "
        "Usual, Customary, and Reasonable (UCR) rate is derived from HealthGuard's contracted fee schedule "
        "for in-network providers or the 80th percentile of the FAIR Health database for out-of-network "
        "services. After determining the allowed amount, the system applies member cost-sharing in the "
        "following order: (1) Annual Deductible - checked against the YTD_DEDUCTIBLE accumulator in the "
        "MEMBER_ACCUM table; (2) Co-payment - flat dollar amount per service category as defined in the "
        "plan's benefit grid; (3) Co-insurance - percentage split applied to the remaining balance after "
        "deductible and co-pay. The final provider payment equals: Payment = Allowed Amount - Deductible "
        "Applied - Co-pay - Co-insurance. Key CICS transactions: HC01 (New Claim Entry), HC02 (Claim "
        "Inquiry/Status), HC03 (Batch Processing Control), HC04 (Provider Fee Schedule Lookup), HC05 "
        "(Member Benefit Inquiry), HC06 (Claim Adjustment/Void)."
    ), y)


def page5_eligibility(doc):
    """Page 5 - Eligibility Verification"""
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    add_header_footer(page, 5, 10)
    y = MARGIN_TOP + 4

    y = insert_heading(page, "4. Eligibility Verification Rules", y, level=1)
    y += 2

    y = insert_body(page, (
        "Eligibility verification is the foundational step in claims adjudication. Before any clinical "
        "or financial evaluation can occur, the system must confirm that the member was actively covered "
        "under a valid HealthGuard policy on the date(s) of service. Eligibility is verified through a "
        "real-time inquiry to the MEMBER_ELIG VSAM master file via CICS transaction HC05. The verification "
        "process evaluates multiple data elements in a cascading validation sequence, and failure at any "
        "checkpoint results in claim rejection or pending status."
    ), y)
    y += 4

    # Insert eligibility image
    y = insert_image(page, IMAGES["eligibility"], y, img_width=CONTENT_W * 0.45)
    y += 2

    y = insert_heading(page, "4.1 Active Policy & Waiting Period Rules", y, level=2)
    y = insert_body(page, (
        "The system first validates the member's policy status. Only policies with STATUS-CODE = 'A' "
        "(Active) or 'C' (COBRA continuation) are eligible for claim payment. Policies in grace period "
        "(STATUS-CODE = 'G') are provisionally eligible, but payment is held pending premium receipt. "
        "Terminated policies (STATUS-CODE = 'T') result in immediate denial with code DN-ELIG-002. "
        "Next, the system evaluates waiting period applicability. HealthGuard enforces a 30-day general "
        "waiting period from the policy effective date during which only emergency and accident-related "
        "claims are payable. A 2-year (730-day) waiting period applies to Pre-Existing Diseases (PED) "
        "declared at the time of enrollment. Specific conditions such as cataract surgery, joint replacement, "
        "and hernia repair carry a mandatory 24-month waiting period regardless of PED status. Maternity "
        "benefits have a separate 9-month waiting period from the policy inception date."
    ), y)
    y += 4

    y = insert_heading(page, "4.2 Age Limits, Sum Insured & Network Verification", y, level=2)
    y = insert_body(page, (
        "HealthGuard policies enforce age-based eligibility limits: dependent children are covered from "
        "birth through age 26 (in compliance with ACA requirements), and primary insureds are eligible for "
        "coverage from age 18 through age 65 (age 70 for renewal without break). Sum insured verification "
        "ensures the claim amount does not exceed the remaining sum insured for the policy year, calculated "
        "as: Remaining SI = Base Sum Insured + Cumulative Bonus - YTD Claims Paid. Sub-limits are enforced "
        "per disease category: room rent is capped at 1% of sum insured per day, ICU charges at 2% of sum "
        "insured per day, and specific procedures (e.g., cataract: $2,500 per eye, knee replacement: "
        "$12,000 per knee) carry per-occurrence caps."
    ), y)
    y += 2
    y = insert_body(page, (
        "Network hospital verification confirms whether the treating facility holds a current contract "
        "with HealthGuard. In-network providers are identified by matching the facility's NPI or "
        "HealthGuard Provider ID against the PROVIDER_NETWORK DB2 table. Out-of-network claims are "
        "processed at reduced benefit levels (typically 60% of UCR versus 80-90% for in-network). "
        "Pre-authorization is mandatory for all planned (elective) hospitalizations at in-network "
        "facilities and must be obtained at least 7 business days prior to the scheduled admission date. "
        "Emergency admissions require notification within 48 hours. Failure to obtain pre-authorization "
        "results in a 20% co-insurance penalty applied on top of normal cost-sharing."
    ), y)


def page6_coverage(doc):
    """Page 6 - Coverage Tiers"""
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    add_header_footer(page, 6, 10)
    y = MARGIN_TOP + 4

    y = insert_heading(page, "5. Coverage Tiers & Benefit Structures", y, level=1)
    y += 2

    y = insert_body(page, (
        "HealthGuard offers four standardized coverage tiers designed to meet varying member needs and "
        "budget levels. Each tier defines the cost-sharing ratio between the plan and the member, commonly "
        "expressed as the actuarial value (AV) - the percentage of total average healthcare costs paid by "
        "the plan. These tiers comply with ACA metal-level requirements and are filed with state insurance "
        "departments in all operating jurisdictions. The following chart illustrates the tier structure."
    ), y)
    y += 4

    # Insert coverage tiers image
    y = insert_image(page, IMAGES["coverage"], y, img_width=CONTENT_W * 0.45)
    y += 2

    y = insert_heading(page, "5.1 Plan Details by Tier", y, level=2)

    # Coverage tier table
    headers = ["Feature", "Bronze (60/40)", "Silver (70/30)", "Gold (80/20)", "Platinum (90/10)"]
    col_w = [CONTENT_W * 0.24, CONTENT_W * 0.19, CONTENT_W * 0.19, CONTENT_W * 0.19, CONTENT_W * 0.19]
    rows = [
        ["Annual Deductible", "$7,000 / $14,000", "$4,500 / $9,000", "$1,500 / $3,000", "$500 / $1,000"],
        ["OOP Maximum", "$9,100 / $18,200", "$8,150 / $16,300", "$6,000 / $12,000", "$3,000 / $6,000"],
        ["PCP Co-pay", "$40", "$30", "$20", "$10"],
        ["Specialist Co-pay", "$80", "$60", "$40", "$20"],
        ["ER Co-pay", "$350", "$250", "$150", "$100"],
        ["Rx Generic Co-pay", "$20", "$15", "$10", "$5"],
        ["Monthly Premium", "$320-$410", "$450-$560", "$590-$720", "$780-$950"],
        ["Preventive Care", "100% covered", "100% covered", "100% covered", "100% covered"],
    ]
    y = draw_table(page, headers, rows, y, col_w)
    y += 4

    y = insert_body(page, (
        "All tiers provide coverage for the ten Essential Health Benefits (EHBs) mandated by the Affordable "
        "Care Act: ambulatory services, emergency services, hospitalization, maternity and newborn care, "
        "mental health and substance use disorder services, prescription drugs, rehabilitative services, "
        "laboratory services, preventive and wellness services, and pediatric services including dental "
        "and vision for children under 19. In-network preventive care services are covered at 100% across "
        "all tiers with no member cost-sharing, in accordance with ACA Section 2713."
    ), y)
    y += 2
    y = insert_body(page, (
        "Mental health and substance abuse services follow federal parity requirements under MHPAEA. "
        "Inpatient mental health carries the same cost-sharing as medical/surgical inpatient admissions "
        "within each tier. Outpatient behavioral health visits are subject to the specialist co-pay "
        "amount. Prescription drug coverage follows a four-tier formulary: Tier 1 (Generics), Tier 2 "
        "(Preferred Brand), Tier 3 (Non-Preferred Brand), Tier 4 (Specialty). Specialty drug co-insurance "
        "ranges from 20% (Platinum) to 40% (Bronze) with a per-script maximum of $250."
    ), y)


def page7_denial(doc):
    """Page 7 - Claim Denial Reasons"""
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    add_header_footer(page, 7, 10)
    y = MARGIN_TOP + 4

    y = insert_heading(page, "6. Claim Denial Reasons & Resolution Procedures", y, level=1)
    y += 2

    y = insert_body(page, (
        "Claim denials represent a critical juncture in the claims lifecycle where the payer determines "
        "that a submitted claim cannot be paid as billed. HealthGuard maintains a denial rate target of "
        "below 8% for clean claims and provides detailed denial reason codes to facilitate resolution. "
        "Each denial triggers an automated Explanation of Benefits (EOB) to the member and a Remittance "
        "Advice (RA) to the provider with the specific denial code and instructions for corrective action."
    ), y)
    y += 4

    # Insert denial reasons image
    y = insert_image(page, IMAGES["denial"], y, img_width=CONTENT_W * 0.45)
    y += 2

    y = insert_heading(page, "6.1 Denial Codes & Resolution Steps", y, level=2)

    denial_items = [
        ("DN001 - Incomplete Documentation:", "Required supporting documents (discharge summary, itemized "
         "bills, investigation reports) were not submitted or are illegible. Resolution: Resubmit with "
         "complete documentation within 45 days of denial notice. Use resubmission code FREQ=7."),
        ("DN002 - Pre-Authorization Missing:", "Planned hospitalization or high-cost procedure was performed "
         "without obtaining prior authorization. Resolution: Submit retroactive authorization request with "
         "clinical justification within 30 days. Approval rate for retroactive auth: 62%."),
        ("DN003 - Policy Exclusion:", "The reported service falls under a contractual exclusion (e.g., "
         "cosmetic surgery, experimental treatments, infertility beyond plan limits). Resolution: Review "
         "policy exclusion list; if clinical necessity overrides exclusion, submit appeal with peer-reviewed "
         "literature supporting medical necessity."),
        ("DN004 - Coding Error:", "ICD-10/CPT code mismatch, invalid modifier usage, or unbundling detected "
         "by NCCI edits. Resolution: Correct coding and resubmit. Common errors include missing modifier 25 "
         "on E&M services performed on the same day as a procedure."),
        ("DN005 - Out-of-Network:", "Services rendered by a non-contracted provider without member consent "
         "to out-of-network cost-sharing. Resolution: Verify network status; if provider was incorrectly "
         "listed, submit network status dispute with directory screenshot evidence."),
        ("DN006 - Timely Filing Exceeded:", "Claim submitted beyond the contractual filing deadline. "
         "Resolution: Submit proof of timely filing (original submission confirmation, EDI 277CA "
         "acknowledgment) or evidence of extraordinary circumstances for filing extension."),
        ("DN007 - Duplicate Claim:", "System detected a matching claim with same member, provider, dates of "
         "service, and procedure codes already on file. Resolution: If not a true duplicate, submit with "
         "documentation of distinct services (different time of day, different anatomical site, modifier 76 "
         "or 77 as appropriate)."),
    ]
    for label, desc in denial_items:
        y = insert_body_bold(page, label, y, indent=4)
        y = insert_body(page, desc, y, indent=10)
        y += 1
        if y > PAGE_H - MARGIN_BOTTOM - 20:
            break


def page8_adjudication(doc):
    """Page 8 - Adjudication Rules Engine"""
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    add_header_footer(page, 8, 10)
    y = MARGIN_TOP + 4

    y = insert_heading(page, "7. Adjudication Rules Engine", y, level=1)
    y += 2

    y = insert_body(page, (
        "HealthGuard's automated adjudication engine is the backbone of claims processing, executing a "
        "rules-based evaluation of every claim that passes initial validation. The engine processes claims "
        "through a priority-ordered rule set, applying each rule sequentially until a terminal disposition "
        "(pay, deny, or pend for manual review) is reached. The rules engine is implemented as a COBOL "
        "rules table (HCADJ-RULES-TBL) within the batch adjudication program HCADJ001 and is updated "
        "quarterly to reflect policy changes, regulatory updates, and medical policy revisions."
    ), y)
    y += 4

    # Insert adjudication image
    y = insert_image(page, IMAGES["adjudication"], y, img_width=CONTENT_W * 0.50)
    y += 2

    y = insert_heading(page, "7.1 Adjudication Rules (R001-R006)", y, level=2)
    rules = [
        ("R001 - Eligibility Check:", "Validates member enrollment status, effective/termination dates, "
         "and benefit plan assignment as of the date of service. Priority: 1 (highest)."),
        ("R002 - Timely Filing Check:", "Verifies the claim was received within the contractual filing "
         "limit (90 days for in-network, 180 days for out-of-network). Priority: 2."),
        ("R003 - Duplicate Detection:", "Compares claim against CLAIMS_HISTORY using member ID, provider "
         "NPI, date of service, CPT code, and modifier to identify exact and near-duplicates. Priority: 3."),
        ("R004 - Coding Validation:", "Executes NCCI edit checks, validates ICD-10/CPT pairing, checks "
         "gender-specific and age-specific procedure appropriateness. Priority: 4."),
        ("R005 - Benefit Application:", "Applies plan-specific benefits including deductible, co-pay, "
         "co-insurance, and sub-limit calculations. Checks annual/lifetime maximums. Priority: 5."),
        ("R006 - Provider Contract:", "Retrieves contracted rates from FEE_SCHEDULE table, applies "
         "contracted discount, and calculates allowed amount per line item. Priority: 6."),
    ]
    for label, desc in rules:
        y = insert_body_bold(page, label, y, indent=4)
        y = insert_body(page, desc, y, indent=10)
        y += 1

    y += 2
    y = insert_heading(page, "7.2 Auto-Adjudication & Manual Review Triggers", y, level=2)
    y = insert_body(page, (
        "Claims meeting all of the following criteria are auto-adjudicated without human intervention: "
        "total billed amount under $500, all CPT-ICD pairs present in the VALID_PAIRS table, rendering "
        "provider is in-network with a current contract, member has no other insurance (no COB), and no "
        "clinical edit flags are triggered. Approximately 72% of HealthGuard claims are auto-adjudicated. "
        "Manual review is triggered for: claims exceeding $5,000 (routed to senior examiner queue), "
        "claims from flagged providers under investigation (SIU review queue), complex procedures "
        "requiring medical director sign-off (clinical review queue), and COB claims requiring "
        "primary/secondary determination. COBOL batch jobs HCADJ001 (nightly adjudication run, "
        "JCL member HCADJJCL in PROD.PROCLIB) and HCADJ002 (weekly reprocessing of pended claims) "
        "handle the bulk of automated processing, with cycle times of approximately 3.5 hours for "
        "the nightly run processing 15,000-22,000 claims."
    ), y)


def page9_appeals(doc):
    """Page 9 - Appeals Process"""
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    add_header_footer(page, 9, 10)
    y = MARGIN_TOP + 4

    y = insert_heading(page, "8. Appeals Process for Denied Claims", y, level=1)
    y += 2

    y = insert_body(page, (
        "HealthGuard provides a comprehensive, multi-level appeals process for members and providers who "
        "disagree with a claim denial or adverse benefit determination. The appeals process is designed in "
        "compliance with ACA Section 2719, ERISA (for employer-sponsored plans), and applicable state "
        "insurance regulations. Members have the right to appeal any denial, and HealthGuard is committed "
        "to a fair, thorough, and timely review of all appeals. In 2023, HealthGuard processed 48,200 "
        "appeals with an overall overturn rate of 41%."
    ), y)
    y += 4

    # Insert appeals process image
    y = insert_image(page, IMAGES["appeals"], y, img_width=CONTENT_W * 0.45)
    y += 2

    y = insert_heading(page, "8.1 Level 1: Internal Appeal", y, level=2)
    y = insert_body(page, (
        "The member or authorized representative must file a written appeal within 180 calendar days of "
        "receiving the adverse benefit determination (EOB denial notice). Appeals should be submitted via "
        "the HealthGuard Member Portal, by mail to PO Box 45200, or by fax to (800) 555-APPL. The appeal "
        "must include the member's name, member ID, claim number, a statement of why the member believes "
        "the denial was incorrect, and any additional supporting documentation (medical records, letters "
        "of medical necessity from the treating physician, peer-reviewed clinical literature). The internal "
        "appeal is reviewed by a Medical Director who was not involved in the original denial decision. "
        "A determination is issued within 30 calendar days for pre-service appeals and 60 calendar days "
        "for post-service appeals. Expedited appeals for urgent or concurrent care situations are decided "
        "within 72 hours. The overturn rate for Level 1 internal appeals was 38% in 2023."
    ), y)
    y += 4

    y = insert_heading(page, "8.2 Level 2: External Review", y, level=2)
    y = insert_body(page, (
        "If the internal appeal is upheld (denied), the member may request an Independent External Review "
        "within 4 months of receiving the Level 1 decision. The request is filed with the applicable State "
        "Insurance Department or, for self-funded ERISA plans, with the Department of Labor. An "
        "Independent Review Organization (IRO) certified by the state conducts the review. The IRO panel "
        "consists of board-certified physicians in the relevant specialty who have no affiliation with "
        "HealthGuard. The IRO's decision is binding on HealthGuard and is issued within 45 calendar days "
        "(72 hours for expedited reviews). The external review overturn rate for HealthGuard cases was "
        "52% in 2023, significantly higher than the industry average of 43%."
    ), y)
    y += 2
    y = insert_body(page, (
        "Required documentation for external review includes: copies of all internal appeal correspondence, "
        "the original claim and EOB, relevant medical records, a letter from the treating physician "
        "detailing clinical rationale, applicable clinical guidelines or medical literature, and the "
        "member's signed authorization for release of medical information to the IRO. Success rates vary "
        "by denial type: medical necessity denials have a 58% overturn rate, coding-related denials 35%, "
        "pre-authorization denials 47%, and policy exclusion denials 22%."
    ), y)


def page10_reference(doc):
    """Page 10 - Reference Tables & Codes"""
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    add_header_footer(page, 10, 10)
    y = MARGIN_TOP + 4

    y = insert_heading(page, "9. Reference Tables & Code Compendium", y, level=1)
    y += 2

    # CPT Codes table
    y = insert_heading(page, "9.1 Common CPT Code Ranges", y, level=2)
    cpt_headers = ["CPT Range", "Category", "Description"]
    cpt_widths = [CONTENT_W * 0.22, CONTENT_W * 0.30, CONTENT_W * 0.48]
    cpt_rows = [
        ["99201 - 99215", "E&M Office Visits", "New & established patient evaluation and management"],
        ["99221 - 99223", "Initial Hospital Care", "Inpatient admission evaluation per day"],
        ["99281 - 99285", "Emergency Dept", "ED visit levels 1 (minor) through 5 (critical)"],
        ["10021 - 10180", "Integumentary", "Incision, drainage, debridement procedures"],
        ["27447", "Musculoskeletal", "Total knee arthroplasty (replacement)"],
        ["33533 - 33536", "Cardiovascular", "Coronary artery bypass graft (CABG)"],
        ["43239", "Digestive", "Upper GI endoscopy with biopsy"],
    ]
    y = draw_table(page, cpt_headers, cpt_rows, y, cpt_widths)
    y += 2

    # ICD-10 Categories
    y = insert_heading(page, "9.2 ICD-10-CM Chapter Categories", y, level=2)
    icd_headers = ["Code Range", "Chapter", "Common Conditions"]
    icd_widths = [CONTENT_W * 0.18, CONTENT_W * 0.28, CONTENT_W * 0.54]
    icd_rows = [
        ["A00 - B99", "Infectious/Parasitic", "COVID-19, pneumonia, sepsis, UTI, cellulitis"],
        ["C00 - D49", "Neoplasms", "Breast cancer, lung cancer, colon cancer, lymphoma"],
        ["E00 - E89", "Endocrine/Metabolic", "Diabetes mellitus (E11.x), hypothyroidism, obesity"],
        ["I00 - I99", "Circulatory", "Hypertension (I10), MI (I21), CHF (I50), AFib (I48)"],
        ["J00 - J99", "Respiratory", "Asthma (J45), COPD (J44), pneumonia (J18)"],
        ["M00 - M99", "Musculoskeletal", "Back pain (M54), osteoarthritis (M17), fractures"],
    ]
    y = draw_table(page, icd_headers, icd_rows, y, icd_widths)
    y += 2

    # Modifier codes
    y = insert_heading(page, "9.3 Common Modifier Codes", y, level=2)
    mod_headers = ["Modifier", "Description", "Usage"]
    mod_widths = [CONTENT_W * 0.12, CONTENT_W * 0.38, CONTENT_W * 0.50]
    mod_rows = [
        ["25", "Significant, separate E&M", "Append to E&M when procedure done same day"],
        ["26", "Professional component", "Physician interpretation only (no technical)"],
        ["59", "Distinct procedural service", "Override NCCI bundling edits when appropriate"],
        ["76", "Repeat procedure, same MD", "Same procedure repeated by same physician"],
        ["77", "Repeat procedure, diff MD", "Same procedure repeated by different physician"],
    ]
    y = draw_table(page, mod_headers, mod_rows, y, mod_widths)
    y += 2

    # Timely Filing & COB
    y = insert_heading(page, "9.4 Timely Filing Limits by Payer Type", y, level=2)
    tf_headers = ["Payer Type", "Filing Limit", "Extension Allowed"]
    tf_widths = [CONTENT_W * 0.33, CONTENT_W * 0.34, CONTENT_W * 0.33]
    tf_rows = [
        ["Commercial (In-Network)", "90 days from DOS", "Up to 180 days with cause"],
        ["Commercial (Out-of-Network)", "180 days from DOS", "Up to 365 days with cause"],
        ["Medicare Advantage", "365 days from DOS", "Per CMS guidelines"],
        ["Medicaid Managed Care", "180 days from DOS", "State-specific rules apply"],
        ["Workers' Compensation", "Per state statute", "Varies by jurisdiction"],
    ]
    y = draw_table(page, tf_headers, tf_rows, y, tf_widths)
    y += 2

    # COB Rules
    y = insert_heading(page, "9.5 Coordination of Benefits (COB) Rules", y, level=2)
    y = insert_body(page, (
        "When a member has dual coverage, HealthGuard follows NAIC Model COB guidelines to determine "
        "primary/secondary payer status. Birthday Rule: For dependent children covered by both parents, "
        "the parent whose birthday falls earlier in the calendar year is primary (month/day only, not "
        "birth year). Gender Rule: Used only in states that have not adopted the Birthday Rule. "
        "Active/Inactive Rule: Coverage through active employment is primary over COBRA, retiree, or "
        "laid-off coverage. Longer/Shorter Rule: If none of the above rules resolve the order, the plan "
        "that has covered the member longer is primary. As secondary payer, HealthGuard pays the lesser "
        "of: (a) its normal benefit amount, or (b) the remaining balance after the primary payer's "
        "payment, ensuring total payments do not exceed 100% of the allowable charges."
    ), y)


# ============================================================================
# MAIN - Build the PDF
# ============================================================================
def main():
    print("Creating PDF document...")
    doc = fitz.open()

    print("  Page 1: Title Page")
    page1_title(doc)

    print("  Page 2: Overview")
    page2_overview(doc)

    print("  Page 3: Claim Submission Process")
    page3_submission(doc)

    print("  Page 4: Mainframe Processing Screen")
    page4_mainframe(doc)

    print("  Page 5: Eligibility Verification")
    page5_eligibility(doc)

    print("  Page 6: Coverage Tiers")
    page6_coverage(doc)

    print("  Page 7: Claim Denial Reasons")
    page7_denial(doc)

    print("  Page 8: Adjudication Rules Engine")
    page8_adjudication(doc)

    print("  Page 9: Appeals Process")
    page9_appeals(doc)

    print("  Page 10: Reference Tables & Codes")
    page10_reference(doc)

    # Save
    os.makedirs(os.path.dirname(OUTPUT_PDF), exist_ok=True)
    doc.save(OUTPUT_PDF)
    doc.close()

    file_size = os.path.getsize(OUTPUT_PDF)
    print(f"\nPDF generated successfully!")
    print(f"  Path: {OUTPUT_PDF}")
    print(f"  Size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
    print(f"  Pages: 10")


if __name__ == "__main__":
    main()
