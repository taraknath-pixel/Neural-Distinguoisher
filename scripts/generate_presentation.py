"""
generate_presentation.py: Creates a clean, minimalist white-and-black presentation
with large typography (body text 18pt+, titles 30-46pt), split across 12 clear slides
to ensure high readability.
"""

import os
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE


def create_deck(output_pptx="Neural_Distinguisher_Flowchart.pptx"):
    prs = Presentation()
    # 16:9 Widescreen layout
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]  # Blank slide

    # Minimalist Palette: Pure White & Solid Black
    WHITE_BG = RGBColor(255, 255, 255)
    CARD_BG = RGBColor(255, 255, 255)
    BORDER_COLOR = RGBColor(40, 40, 40)
    HEADER_FILL = RGBColor(245, 245, 245)
    BLACK_TEXT = RGBColor(0, 0, 0)
    SUB_TEXT = RGBColor(50, 50, 50)
    MUTED_TEXT = RGBColor(80, 80, 80)

    def set_slide_background(slide):
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
        bg.fill.solid()
        bg.fill.fore_color.rgb = WHITE_BG
        bg.line.fill.background()

    def add_header(slide, title_text, category_text="NEURAL DISTINGUISHER PIPELINE"):
        # Category tag (14pt)
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.35))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(14)
        p_cat.font.bold = True
        p_cat.font.color.rgb = MUTED_TEXT

        # Main Title (30pt)
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.75), Inches(11.7), Inches(0.7))
        tf_title = title_box.text_frame
        tf_title.word_wrap = True
        p_title = tf_title.paragraphs[0]
        p_title.text = title_text
        p_title.font.size = Pt(30)
        p_title.font.bold = True
        p_title.font.color.rgb = BLACK_TEXT

        # Dividing line
        line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.5), Inches(11.7), Inches(0.03))
        line.fill.solid()
        line.fill.fore_color.rgb = BORDER_COLOR
        line.line.fill.background()

    # =========================================================================
    # SLIDE 1: Title Slide
    # =========================================================================
    slide1 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide1)

    bar = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.2), Inches(3.0), Inches(0.08))
    bar.fill.solid()
    bar.fill.fore_color.rgb = BLACK_TEXT
    bar.line.fill.background()

    tbox = slide1.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.7), Inches(4.5))
    tf1 = tbox.text_frame
    tf1.word_wrap = True

    p0 = tf1.paragraphs[0]
    p0.text = "Neural Distinguisher"
    p0.font.size = Pt(48)
    p0.font.bold = True
    p0.font.color.rgb = BLACK_TEXT

    p1 = tf1.add_paragraph()
    p1.text = "Cryptographic Algorithm Identification using Deep Learning"
    p1.font.size = Pt(26)
    p1.font.bold = True
    p1.font.color.rgb = SUB_TEXT
    p1.space_before = Pt(12)

    p2 = tf1.add_paragraph()
    p2.text = "End-to-End Pipeline Architecture: Simulation, 49 NIST Feature Extraction, Multi-Architecture Classification, and Interactive User Inference."
    p2.font.size = Pt(19)
    p2.font.color.rgb = MUTED_TEXT
    p2.space_before = Pt(20)

    # =========================================================================
    # SLIDE 2: Executive Overview (The 3 Key Pillars)
    # =========================================================================
    slide2 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide2)
    add_header(slide2, "Executive Overview: Core System Pillars", "SYSTEM ARCHITECTURE")

    pillars = [
        ("92.0%", "Peak 1D CNN Accuracy", "Evaluated on 512 KB AES vs. 3DES ciphertexts, outperforming traditional ML models by up to 38%."),
        ("49 Features", "NIST SP 800-22 Tests", "Comprehensive suite of bit, block, spectral, and complexity statistical tests extracted per sample."),
        ("25,000 Samples", "Empirical Dataset", "Balanced dataset covering 5 block ciphers across 5 file sizes with 1,000 samples each.")
    ]

    for i, (metric, label, desc) in enumerate(pillars):
        x = Inches(0.8 + i * 4.0)
        card = slide2.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, Inches(1.9), Inches(3.7), Inches(4.8))
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = BORDER_COLOR
        card.line.width = Pt(2)

        ctf = card.text_frame
        ctf.word_wrap = True

        p0 = ctf.paragraphs[0]
        p0.text = metric
        p0.font.size = Pt(40)
        p0.font.bold = True
        p0.font.color.rgb = BLACK_TEXT
        p0.alignment = PP_ALIGN.CENTER

        p1 = ctf.add_paragraph()
        p1.text = label
        p1.font.size = Pt(20)
        p1.font.bold = True
        p1.font.color.rgb = BLACK_TEXT
        p1.alignment = PP_ALIGN.CENTER
        p1.space_before = Pt(10)

        p2 = ctf.add_paragraph()
        p2.text = desc
        p2.font.size = Pt(18)
        p2.font.color.rgb = SUB_TEXT
        p2.space_before = Pt(16)

    # =========================================================================
    # SLIDE 3: Master System Flowchart (Stages 1 to 5)
    # =========================================================================
    slide3 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide3)
    add_header(slide3, "System Flowchart: End-to-End Execution Flow", "PIPELINE WORKFLOW")

    flow_stages = [
        ("Stage 1", "Simulation & Data Gen", "data_generation.py", "25,000 ciphertext samples generated across 5 ciphers & 5 block sizes."),
        ("Stage 2", "NIST Feature Extraction", "feature_extraction.py", "49 statistical randomness metrics computed per ciphertext sample."),
        ("Stage 3", "Multi-Architecture Training", "classification.py", "1D CNN, MLP, and 5 traditional baseline ML classifiers trained & tuned."),
        ("Stage 4", "Model Calibration & Loading", "inference.py / models", "Decision thresholds calibrated; lightweight model weights packaged."),
        ("Stage 5", "Tester & Live User Inference", "predict.py / api.py", "Demo-1 empirical evaluation and Demo-2 live normalizer (Hex, CSV, System).")
    ]

    for i, (stg_num, stg_name, script_name, stg_desc) in enumerate(flow_stages):
        y = Inches(1.8 + i * 1.0)
        card = slide3.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), y, Inches(11.7), Inches(0.85))
        card.fill.solid()
        card.fill.fore_color.rgb = CARD_BG
        card.line.color.rgb = BORDER_COLOR
        card.line.width = Pt(1.5)

        ctf = card.text_frame
        ctf.vertical_anchor = MSO_ANCHOR.MIDDLE
        ctf.word_wrap = True

        p = ctf.paragraphs[0]
        p.text = f"{stg_num}: {stg_name}  ({script_name})"
        p.font.size = Pt(19)
        p.font.bold = True
        p.font.color.rgb = BLACK_TEXT

        p2 = ctf.add_paragraph()
        p2.text = stg_desc
        p2.font.size = Pt(18)
        p2.font.color.rgb = SUB_TEXT

    # =========================================================================
    # SLIDE 4: Stage 1 — Simulation & Data Generation
    # =========================================================================
    slide4 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide4)
    add_header(slide4, "Stage 1: Ciphertext Simulation Methodology", "STAGE 1 — DATA GENERATION")

    cbox4 = slide4.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf4 = cbox4.text_frame
    tf4.word_wrap = True

    pts4 = [
        ("High-Entropy Plaintext Generation",
         "Constructed by concatenating 16-byte random UUID4 blocks until the target size (1 KB to 512 KB) is reached, simulating realistic structured data."),
        ("ECB Encryption Mode with Fixed Keys",
         "Plaintexts are encrypted in Electronic Codebook (ECB) mode to isolate intrinsic diffusion behavior across cryptographic algorithms without IV masking."),
        ("Strict PKCS#7 Padding Alignment",
         "128-bit block padding for AES, and 64-bit block padding for 3DES, CAST-128, RC2, and Blowfish to maintain valid cryptographic standards."),
        ("Automated Directory Partitioning",
         "Organized as crypto_validation_dataset/ciphertext/<Algorithm>/<Size>/sample_<ID>.bin ready for feature extraction.")
    ]

    for title, desc in pts4:
        p = tf4.add_paragraph()
        p.text = f"• {title}:"
        p.font.size = Pt(20)
        p.font.bold = True
        p.font.color.rgb = BLACK_TEXT
        p.space_before = Pt(10)

        pd = tf4.add_paragraph()
        pd.text = desc
        pd.font.size = Pt(18)
        pd.font.color.rgb = SUB_TEXT
        pd.space_before = Pt(2)

    # =========================================================================
    # SLIDE 5: Stage 1 — The 25,000 Sample Empirical Grid
    # =========================================================================
    slide5 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide5)
    add_header(slide5, "Stage 1: Complete 25,000 Sample Dataset Grid", "STAGE 1 — DATASET GRID")

    sub_box5 = slide5.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.7), Inches(0.5))
    p_sub5 = sub_box5.text_frame.paragraphs[0]
    p_sub5.text = "Mathematical Grid: 5 Ciphers × 5 File Sizes × 1,000 Samples = 25,000 Total Files"
    p_sub5.font.size = Pt(19)
    p_sub5.font.bold = True
    p_sub5.font.color.rgb = SUB_TEXT

    # Large Table (18pt text)
    rows, cols = 7, 6
    tbl5_shape = slide5.shapes.add_table(rows, cols, Inches(0.8), Inches(2.2), Inches(11.7), Inches(4.6))
    table5 = tbl5_shape.table
    table5.columns[0].width = Inches(2.7)
    for c in range(1, 6):
        table5.columns[c].width = Inches(1.8)

    headers5 = ["Algorithm", "1 KB", "8 KB", "64 KB", "256 KB", "512 KB"]
    for c, h in enumerate(headers5):
        cell = table5.cell(0, c)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = HEADER_FILL
        p = cell.text_frame.paragraphs[0]
        p.font.size = Pt(18)
        p.font.bold = True
        p.font.color.rgb = BLACK_TEXT
        p.alignment = PP_ALIGN.CENTER

    data5 = [
        ("AES", "1,000", "1,000", "1,000", "1,000", "1,000"),
        ("3DES", "1,000", "1,000", "1,000", "1,000", "1,000"),
        ("CAST-128", "1,000", "1,000", "1,000", "1,000", "1,000"),
        ("RC2", "1,000", "1,000", "1,000", "1,000", "1,000"),
        ("Blowfish", "1,000", "1,000", "1,000", "1,000", "1,000"),
        ("Total per Size", "5,000", "5,000", "5,000", "5,000", "5,000")
    ]
    for r, row in enumerate(data5):
        for c, val in enumerate(row):
            cell = table5.cell(r + 1, c)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = HEADER_FILL if r == 5 else WHITE_BG
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(18)
            p.font.bold = (r == 5 or c == 0)
            p.font.color.rgb = BLACK_TEXT
            p.alignment = PP_ALIGN.CENTER

    # =========================================================================
    # SLIDE 6: Stage 2 — NIST SP 800-22 Feature Extraction
    # =========================================================================
    slide6 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide6)
    add_header(slide6, "Stage 2: NIST SP 800-22 Feature Suite (49 Features)", "STAGE 2 — FEATURE EXTRACTION")

    cbox6 = slide6.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf6 = cbox6.text_frame
    tf6.word_wrap = True

    pts6 = [
        ("Bit & Block Level Tests (9 Features)",
         "Monobit Frequency, Block Frequency (M=128), Runs Test, Proportion of Ones, Longest Run of Ones, and Binary Matrix Rank (32×32)."),
        ("Spectral & Non-Overlapping Template Tests (33 Features)",
         "Discrete Fourier Transform (Spectral peak detection) and 39 aperiodic non-overlapping template matching tests detecting non-random bit patterns."),
        ("Complexity & Cumulative Walk Tests (7 Features)",
         "Maurer's Universal Statistical Test, Approximate Entropy (m=10), Forward/Reverse Cumulative Sums, and Random Excursion distribution tests."),
        ("Extracted Feature Matrix",
         "Constructs a unified, normalized matrix of shape (25000, 49) stored in extracted_features.npz for ML consumption.")
    ]

    for title, desc in pts6:
        p = tf6.add_paragraph()
        p.text = f"• {title}:"
        p.font.size = Pt(20)
        p.font.bold = True
        p.font.color.rgb = BLACK_TEXT
        p.space_before = Pt(10)

        pd = tf6.add_paragraph()
        pd.text = desc
        pd.font.size = Pt(18)
        pd.font.color.rgb = SUB_TEXT
        pd.space_before = Pt(2)

    # =========================================================================
    # SLIDE 7: Stage 2 — Engineering Optimization (1,500x Speedup)
    # =========================================================================
    slide7 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide7)
    add_header(slide7, "Stage 2: 1,500x Vectorization Speedup", "ENGINEERING ACCELERATION")

    cbox7 = slide7.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf7 = cbox7.text_frame
    tf7.word_wrap = True

    pts7 = [
        ("The Computational Bottleneck",
         "The standard Python NIST template matching test processes 39 templates across 16,384 blocks = 638,976 nested loop iterations per 512 KB sample. This took ~40 seconds per file, causing pipeline stalls on large inputs."),
        ("NumPy 2D Block Sliding Window Vectorization",
         "Re-engineered non_overlapping_template_matching_test into batch_non_overlapping_template_matching using vectorized 2D strided array operations. This replaces slow interpreted loops with native C-speed matrix broadcasting."),
        ("Empirical Performance Leap",
         "Latency dropped from 40.0 seconds down to 0.023 seconds per sample — a 1,500x speedup that makes manual 512 KB inference instantaneous.")
    ]

    for title, desc in pts7:
        p = tf7.add_paragraph()
        p.text = f"• {title}:"
        p.font.size = Pt(20)
        p.font.bold = True
        p.font.color.rgb = BLACK_TEXT
        p.space_before = Pt(12)

        pd = tf7.add_paragraph()
        pd.text = desc
        pd.font.size = Pt(18)
        pd.font.color.rgb = SUB_TEXT
        pd.space_before = Pt(3)

    # =========================================================================
    # SLIDE 8: Stage 3 — Proposed 1D CNN Architecture
    # =========================================================================
    slide8 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide8)
    add_header(slide8, "Stage 3: Proposed 1D CNN Architecture (PyTorch)", "STAGE 3 — DEEP LEARNING")

    cbox8 = slide8.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf8 = cbox8.text_frame
    tf8.word_wrap = True

    pts8 = [
        ("Input Representation",
         "49-element normalized NIST feature vector reshaped into a 1×49 1D tensor."),
        ("Convolutional Feature Extractor",
         "Conv1D(1 -> 32, kernel=3, pad=1) + BatchNorm1D + ReLU + MaxPool1D(2), followed by Conv1D(32 -> 64, kernel=3, pad=1) + BatchNorm1D + ReLU + MaxPool1D(2)."),
        ("Dense Representation & Regularization",
         "Flattened representation (64 × 12 = 768) projected to Linear(768, 128) + ReLU + Dropout(0.3) to prevent overfitting."),
        ("Classification Head & Optimization",
         "Linear(128, 2) for AES vs. 3DES binary distinction. Trained using Adam optimizer, CrossEntropyLoss, and Early Stopping (patience=50).")
    ]

    for title, desc in pts8:
        p = tf8.add_paragraph()
        p.text = f"• {title}:"
        p.font.size = Pt(20)
        p.font.bold = True
        p.font.color.rgb = BLACK_TEXT
        p.space_before = Pt(10)

        pd = tf8.add_paragraph()
        pd.text = desc
        pd.font.size = Pt(18)
        pd.font.color.rgb = SUB_TEXT
        pd.space_before = Pt(2)

    # =========================================================================
    # SLIDE 9: Stage 3 — The 6 Benchmark ML Baseline Classifiers
    # =========================================================================
    slide9 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide9)
    add_header(slide9, "Stage 3: Benchmark Machine Learning Baselines", "STAGE 3 — BASELINE MODELS")

    cbox9 = slide9.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf9 = cbox9.text_frame
    tf9.word_wrap = True

    pts9 = [
        ("Multi-Layer Perceptron (MLP)",
         "Deep baseline with 2 hidden layers (64 × 32), ReLU activation, Adam optimizer, achieving 82.5% on 512 KB."),
        ("Random Forest (RF)",
         "Ensemble of 100 decision trees with Gini impurity splitting; robust baseline achieving up to 65% accuracy."),
        ("Support Vector Machine (SVM)",
         "Nonlinear Support Vector Classifier using Radial Basis Function (RBF) kernel with Platt probability scaling."),
        ("K-Nearest Neighbors (KNN)",
         "Instance-based classifier (k=5 neighbors) with Euclidean distance metric weighting."),
        ("Logistic Regression (LR) & Gaussian Naive Bayes (GNB)",
         "L2-regularized linear model and probabilistic feature-independence baselines (50% to 62% accuracy).")
    ]

    for title, desc in pts9:
        p = tf9.add_paragraph()
        p.text = f"• {title}:"
        p.font.size = Pt(20)
        p.font.bold = True
        p.font.color.rgb = BLACK_TEXT
        p.space_before = Pt(8)

        pd = tf9.add_paragraph()
        pd.text = desc
        pd.font.size = Pt(18)
        pd.font.color.rgb = SUB_TEXT
        pd.space_before = Pt(2)

    # =========================================================================
    # SLIDE 10: Stage 4 — Empirical Benchmark Results (Chart)
    # =========================================================================
    slide10 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide10)
    add_header(slide10, "Stage 4: Empirical Benchmark Visualization", "STAGE 4 — BENCHMARK EVALUATION")

    chart_path = Path("benchmark_comparison.png")
    if chart_path.exists():
        slide10.shapes.add_picture(str(chart_path), Inches(0.8), Inches(1.8), width=Inches(7.6))

    cbox10 = slide10.shapes.add_textbox(Inches(8.7), Inches(1.8), Inches(3.8), Inches(5.0))
    tf10 = cbox10.text_frame
    tf10.word_wrap = True

    p0 = tf10.paragraphs[0]
    p0.text = "Key Takeaways:"
    p0.font.size = Pt(22)
    p0.font.bold = True
    p0.font.color.rgb = BLACK_TEXT

    insights = [
        "1D CNN reaches 92.0% accuracy on 512 KB, outperforming traditional ML baselines by up to 38%.",
        "Deep convolutional filters successfully discover subtle residual diffusion artifacts across ciphertext blocks.",
        "Traditional ML models plateau around 55% to 65% because handcrafted features lose nonlinear correlations."
    ]
    for insight in insights:
        p = tf10.add_paragraph()
        p.text = f"• {insight}"
        p.font.size = Pt(18)
        p.font.color.rgb = SUB_TEXT
        p.space_before = Pt(12)

    # =========================================================================
    # SLIDE 11: Stage 4 — Empirical Accuracy Comparison Table
    # =========================================================================
    slide11 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide11)
    add_header(slide11, "Stage 4: Accuracy Comparison Matrix (AES vs. 3DES)", "STAGE 4 — ACCURACY TABLE")

    sub_box11 = slide11.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.7), Inches(0.5))
    p_sub11 = sub_box11.text_frame.paragraphs[0]
    p_sub11.text = "Binary Classification Accuracy across File Sizes (10-Fold Stratified Cross-Validation)"
    p_sub11.font.size = Pt(19)
    p_sub11.font.bold = True
    p_sub11.font.color.rgb = SUB_TEXT

    rows, cols = 6, 8
    tbl11_shape = slide11.shapes.add_table(rows, cols, Inches(0.8), Inches(2.2), Inches(11.7), Inches(4.6))
    table11 = tbl11_shape.table
    table11.columns[0].width = Inches(1.6)
    table11.columns[1].width = Inches(1.8)
    for c in range(2, 8):
        table11.columns[c].width = Inches(1.4)

    headers11 = ["Size", "CNN (Ours)", "MLP", "RF", "SVM", "KNN", "LR", "GNB"]
    for c, h in enumerate(headers11):
        cell = table11.cell(0, c)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = HEADER_FILL
        p = cell.text_frame.paragraphs[0]
        p.font.size = Pt(18)
        p.font.bold = True
        p.font.color.rgb = BLACK_TEXT
        p.alignment = PP_ALIGN.CENTER

    data11 = [
        ("1 KB",   "82.0%", "72.5%", "52.5%", "50.0%", "55.0%", "40.0%", "44.0%"),
        ("8 KB",   "84.5%", "72.5%", "57.5%", "52.5%", "52.5%", "50.0%", "52.0%"),
        ("64 KB",  "89.5%", "77.5%", "65.0%", "62.5%", "60.0%", "60.0%", "60.0%"),
        ("256 KB", "89.5%", "77.5%", "62.5%", "57.5%", "57.5%", "54.0%", "62.0%"),
        ("512 KB", "92.0%", "82.5%", "60.0%", "60.0%", "60.0%", "54.0%", "60.0%"),
    ]
    for r, row in enumerate(data11):
        for c, val in enumerate(row):
            cell = table11.cell(r + 1, c)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = WHITE_BG
            p = cell.text_frame.paragraphs[0]
            p.font.size = Pt(18)
            p.font.bold = (c <= 1)
            p.font.color.rgb = BLACK_TEXT
            p.alignment = PP_ALIGN.CENTER

    # =========================================================================
    # SLIDE 12: Stage 5 — Live User Experience (Demo-1 vs. Demo-2)
    # =========================================================================
    slide12 = prs.slides.add_slide(blank_layout)
    set_slide_background(slide12)
    add_header(slide12, "Stage 5: Live User Experience & Inference (Demo-1 vs. Demo-2)", "STAGE 5 — USER EXPERIENCE")

    cbox12 = slide12.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(11.7), Inches(5.0))
    tf12 = cbox12.text_frame
    tf12.word_wrap = True

    pts12 = [
        ("Demo-1: Automated Benchmark Verification Flow",
         "Executes Flowchart Steps 4 & 5 on application launch: verifies 1,000-sample trained models across all 7 architectures, prints empirical accuracy matrix, and automatically pops up the comparison chart."),
        ("Demo-2: Live User Input Normalizer (3 Input Modes)",
         "Mode 1: Manual Hex Input (terminal paste or path to .hex/.txt file with clean 32-byte preview). Mode 2: Batch CSV File Upload (consensus voting). Mode 3: System Generator (encrypts AES or 3DES ciphertext on the fly)."),
        ("Smart Auto-Snap & Size Guardrails",
         "Enforces minimum 1 KB and maximum 512 KB input boundaries. Automatically snaps arbitrary byte lengths to the nearest trained block size (1 KB, 8 KB, 64 KB, 256 KB, 512 KB)."),
        ("Out-of-Distribution (OOD) Rejection",
         "Entropy-based softmax rejection: rejects non-target inputs with 'Does not belong to AES or 3DES' rather than guessing incorrectly.")
    ]

    for title, desc in pts12:
        p = tf12.add_paragraph()
        p.text = f"• {title}:"
        p.font.size = Pt(20)
        p.font.bold = True
        p.font.color.rgb = BLACK_TEXT
        p.space_before = Pt(8)

        pd = tf12.add_paragraph()
        pd.text = desc
        pd.font.size = Pt(18)
        pd.font.color.rgb = SUB_TEXT
        pd.space_before = Pt(2)

    # Save presentation
    prs.save(output_pptx)
    print(f"[+] Presentation successfully created: {os.path.abspath(output_pptx)}")
    return output_pptx


if __name__ == "__main__":
    create_deck("Neural_Distinguisher_Flowchart.pptx")
