#!/usr/bin/env python3
"""
Generate all publication-quality figures for the RetinaSense IEEE paper.
Output: paper/figures/*.png (300 DPI, white background)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Consistent style ──────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Color palette (IEEE-friendly, works in grayscale)
C_BLUE    = '#2171B5'
C_ORANGE  = '#D94801'
C_GREEN   = '#238B45'
C_PURPLE  = '#6A51A3'
C_RED     = '#CB181D'
C_GRAY    = '#636363'
C_LTBLUE  = '#C6DBEF'
C_LTORANGE= '#FDD0A2'
C_LTGREEN = '#C7E9C0'
C_LTPURPLE= '#DADAEB'
C_LTRED   = '#FCBBA1'

CLASS_COLORS = [C_BLUE, C_ORANGE, C_GREEN, C_PURPLE, C_RED]
CLASS_NAMES  = ['Normal', 'DR', 'Glaucoma', 'Cataract', 'AMD']


# ══════════════════════════════════════════════════════════════════
# FIGURE 1: System Architecture (End-to-End Pipeline)
# ══════════════════════════════════════════════════════════════════
def draw_system_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-1.5, 4.5)
    ax.axis('off')

    def draw_box(x, y, w, h, label, color, sublabel=None, fontsize=9, bold=True):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                             facecolor=color, edgecolor='#333333', linewidth=1.2)
        ax.add_patch(box)
        weight = 'bold' if bold else 'normal'
        if sublabel:
            ax.text(x + w/2, y + h/2 + 0.15, label, ha='center', va='center',
                    fontsize=fontsize, fontweight=weight, color='white')
            ax.text(x + w/2, y + h/2 - 0.2, sublabel, ha='center', va='center',
                    fontsize=fontsize - 2, color='white', style='italic')
        else:
            ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                    fontsize=fontsize, fontweight=weight, color='white')

    def draw_arrow(x1, y1, x2, y2, label=None):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#333333',
                                    lw=1.5, connectionstyle='arc3,rad=0'))
        if label:
            mx, my = (x1 + x2)/2, (y1 + y2)/2 + 0.15
            ax.text(mx, my, label, ha='center', va='bottom', fontsize=7,
                    color=C_GRAY, style='italic')

    # ── Main pipeline boxes ──
    y_main = 2.5
    h = 1.0
    # Stage 1: Input
    draw_box(0, y_main, 1.3, h, 'Fundus\nImage', '#888888', fontsize=9)
    # Stage 2: Preprocessing
    draw_box(1.8, y_main, 1.5, h, 'CLAHE\nPreprocessing', C_BLUE, fontsize=9)
    # Stage 3: DANN-ViT
    draw_box(3.8, y_main, 1.8, h, 'DANN-ViT\nBackbone', C_ORANGE, sublabel='86M params', fontsize=9)
    # Stage 4: Calibration
    draw_box(6.1, y_main, 1.5, h, 'Temperature\nCalibration', C_GREEN, fontsize=9)
    # Stage 5: RAD
    draw_box(8.1, y_main, 1.8, h, 'RAD\nPipeline', C_PURPLE, fontsize=9)

    # ── Main arrows ──
    draw_arrow(1.3, y_main + h/2, 1.8, y_main + h/2, '224×224')
    draw_arrow(3.3, y_main + h/2, 3.8, y_main + h/2)
    draw_arrow(5.6, y_main + h/2, 6.1, y_main + h/2, 'logits')
    draw_arrow(7.6, y_main + h/2, 8.1, y_main + h/2)

    # ── Sub-components below DANN-ViT ──
    y_sub = 0.8
    h_sub = 0.7
    draw_box(3.1, y_sub, 1.1, h_sub, 'Disease\nHead', C_ORANGE, fontsize=7, bold=False)
    draw_box(4.3, y_sub, 1.1, h_sub, 'Severity\nHead', C_ORANGE, fontsize=7, bold=False)
    draw_box(5.5, y_sub, 1.1, h_sub, 'Domain\nHead + GRL', C_RED, fontsize=7, bold=False)

    # Arrows down from backbone
    ax.annotate('', xy=(3.65, y_sub + h_sub), xytext=(4.4, y_main),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1, ls='--'))
    ax.annotate('', xy=(4.85, y_sub + h_sub), xytext=(4.7, y_main),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1, ls='--'))
    ax.annotate('', xy=(6.05, y_sub + h_sub), xytext=(5.0, y_main),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1, ls='--'))

    # ── Sub-components below RAD ──
    y_sub2 = 0.8
    draw_box(7.6, y_sub2 + 0.8, 1.0, 0.55, 'MC\nDropout', C_PURPLE, fontsize=7, bold=False)
    draw_box(8.7, y_sub2 + 0.8, 1.0, 0.55, 'FAISS\nRetrieval', C_PURPLE, fontsize=7, bold=False)

    ax.annotate('', xy=(8.1, y_sub2 + 1.35), xytext=(8.7, y_main),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1, ls='--'))
    ax.annotate('', xy=(9.2, y_sub2 + 1.35), xytext=(9.3, y_main),
                arrowprops=dict(arrowstyle='->', color='#555', lw=1, ls='--'))

    # ── Output boxes on top ──
    y_out = 4.0
    h_out = 0.4
    for i, (lbl, col) in enumerate([('Auto-Report', C_GREEN), ('Review', '#D4A017'), ('Escalate', C_RED)]):
        x = 7.6 + i * 1.1
        box = FancyBboxPatch((x, y_out), 0.95, h_out, boxstyle="round,pad=0.08",
                             facecolor=col, edgecolor='#333', linewidth=1)
        ax.add_patch(box)
        ax.text(x + 0.475, y_out + h_out/2, lbl, ha='center', va='center',
                fontsize=7, fontweight='bold', color='white')

    # Arrow from RAD to outputs
    ax.annotate('', xy=(8.55, y_out), xytext=(8.8, y_main + h),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.2))
    ax.annotate('', xy=(9.65, y_out), xytext=(9.2, y_main + h),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.2))
    ax.annotate('', xy=(7.95, y_out), xytext=(8.5, y_main + h),
                arrowprops=dict(arrowstyle='->', color='#333', lw=1.2))

    # ── Preprocessing sub-steps ──
    y_pre = 0.2
    steps = ['Crop\nBorders', 'Resize\n224×224', 'CLAHE\nL-channel', 'Circle\nMask', 'Normalize']
    for i, s in enumerate(steps):
        x = 0.7 + i * 0.65
        box = FancyBboxPatch((x, y_pre), 0.55, 0.5, boxstyle="round,pad=0.06",
                             facecolor=C_LTBLUE, edgecolor=C_BLUE, linewidth=0.8)
        ax.add_patch(box)
        ax.text(x + 0.275, y_pre + 0.25, s, ha='center', va='center',
                fontsize=5.5, color=C_BLUE)

    # Bracket from preprocessing box
    ax.annotate('', xy=(2.2, y_pre + 0.5), xytext=(2.55, y_main),
                arrowprops=dict(arrowstyle='->', color=C_BLUE, lw=0.8, ls=':'))

    # Title
    ax.text(5.0, 4.3, 'RetinaSense: End-to-End Pipeline', ha='center',
            fontsize=13, fontweight='bold', color='#222')

    fig.savefig(os.path.join(OUT_DIR, 'fig_system_architecture.png'))
    plt.close(fig)
    print("  [1/7] fig_system_architecture.png")


# ══════════════════════════════════════════════════════════════════
# FIGURE 2: DANN-ViT Architecture
# ══════════════════════════════════════════════════════════════════
def draw_dann_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.set_xlim(-0.5, 11)
    ax.set_ylim(-0.5, 5.5)
    ax.axis('off')

    def rbox(x, y, w, h, label, fc, ec='#333', fs=8, sublabel=None, alpha=1.0):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                             facecolor=fc, edgecolor=ec, linewidth=1.2, alpha=alpha)
        ax.add_patch(box)
        hx = fc.lstrip('#')
        if len(hx) == 3:
            hx = ''.join(c*2 for c in hx)
        tc = 'white' if sum(int(hx[i:i+2], 16) for i in (0,2,4)) < 400 else '#222'
        if sublabel:
            ax.text(x+w/2, y+h/2+0.15, label, ha='center', va='center',
                    fontsize=fs, fontweight='bold', color=tc)
            ax.text(x+w/2, y+h/2-0.2, sublabel, ha='center', va='center',
                    fontsize=fs-2, color=tc, style='italic')
        else:
            ax.text(x+w/2, y+h/2, label, ha='center', va='center',
                    fontsize=fs, fontweight='bold', color=tc)

    def arrow(x1, y1, x2, y2, lbl=None, color='#333', lw=1.5, style='-'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw, ls=style))
        if lbl:
            mx, my = (x1+x2)/2, max(y1, y2) + 0.15
            ax.text(mx, my, lbl, ha='center', va='bottom', fontsize=7, color=C_GRAY)

    # ── Input ──
    rbox(0, 2.5, 1.2, 1.0, 'Input\n224×224×3', '#AAAAAA', fs=8)

    # ── Patch Embedding ──
    rbox(1.7, 2.5, 1.5, 1.0, 'Patch\nEmbedding', C_LTBLUE, ec=C_BLUE, fs=8)
    ax.text(2.45, 2.35, '196 patches + CLS', ha='center', fontsize=6, color=C_BLUE)
    arrow(1.2, 3.0, 1.7, 3.0, '16×16 patches')

    # ── Transformer blocks ──
    for i in range(4):
        x = 3.7 + i * 0.45
        alpha = 0.5 + 0.15 * i
        rbox(x, 2.5, 0.35, 1.0, '', C_ORANGE, alpha=alpha)
    ax.text(4.35, 3.0, '12× Transformer\nBlocks', ha='center', va='center',
            fontsize=8, fontweight='bold', color='white',
            path_effects=[pe.withStroke(linewidth=2, foreground='#333')])
    ax.text(4.35, 2.35, 'MHSA + MLP + LN', ha='center', fontsize=6, color=C_ORANGE)
    arrow(3.2, 3.0, 3.7, 3.0)
    arrow(5.47, 3.0, 6.0, 3.0, '768-d CLS')

    # ── CLS token box ──
    rbox(6.0, 2.6, 1.0, 0.8, 'CLS\nToken', '#555555', fs=9)

    # ── Three heads ──
    # Disease Head
    y_d = 4.2
    rbox(6.8, y_d, 1.8, 0.9, 'Disease Head', C_BLUE, fs=9)
    ax.text(7.7, y_d - 0.15, '768→512→256→5', ha='center', fontsize=6, color=C_BLUE)
    arrow(6.5, 3.4, 7.1, y_d, color=C_BLUE)

    # Severity Head
    rbox(8.9, y_d, 1.6, 0.9, 'Severity Head', C_GREEN, fs=9)
    ax.text(9.7, y_d - 0.15, '768→256→5', ha='center', fontsize=6, color=C_GREEN)
    arrow(6.8, 3.3, 9.3, y_d, color=C_GREEN)

    # Domain Head with GRL
    y_dom = 0.5
    rbox(7.3, y_dom + 0.95, 0.9, 0.7, 'GRL', C_RED, fs=9)
    ax.text(7.75, y_dom + 0.8, 'λ-scheduled', ha='center', fontsize=6, color=C_RED)
    rbox(8.5, y_dom + 0.6, 1.8, 1.3, 'Domain\nDiscriminator', C_RED, fs=9)
    ax.text(9.4, y_dom + 0.45, '768→256→64→4', ha='center', fontsize=6, color=C_RED)
    arrow(6.5, 2.6, 7.55, y_dom + 1.65, color=C_RED)
    arrow(8.2, y_dom + 1.3, 8.5, y_dom + 1.3, color=C_RED)

    # ── Gradient reversal annotation ──
    ax.annotate('Reversed\nGradients', xy=(7.75, y_dom + 0.95), xytext=(6.3, y_dom + 0.5),
                fontsize=7, color=C_RED, ha='center',
                arrowprops=dict(arrowstyle='->', color=C_RED, lw=1, ls='--'))

    # ── Output labels ──
    rbox(9.0, y_d + 1.0, 1.5, 0.4, '5 Disease Classes', C_LTBLUE, ec=C_BLUE, fs=7)
    arrow(7.7, y_d + 0.9, 9.4, y_d + 1.0, color=C_BLUE, lw=1)

    ax.text(10.5, y_dom + 1.25, '4 Domains', ha='center', fontsize=8,
            fontweight='bold', color=C_RED)
    arrow(10.3, y_dom + 1.3, 10.5, y_dom + 1.15, color=C_RED, lw=1)

    # ── LLRD annotation ──
    ax.annotate('', xy=(3.7, 2.3), xytext=(5.47, 2.3),
                arrowprops=dict(arrowstyle='<->', color=C_GRAY, lw=1))
    ax.text(4.58, 2.05, 'LLRD (decay=0.85)', ha='center', fontsize=7, color=C_GRAY)

    # Title
    ax.text(5.25, 5.2, 'DANN-ViT Architecture', ha='center',
            fontsize=13, fontweight='bold', color='#222')

    fig.savefig(os.path.join(OUT_DIR, 'fig_dann_architecture.png'))
    plt.close(fig)
    print("  [2/7] fig_dann_architecture.png")


# ══════════════════════════════════════════════════════════════════
# FIGURE 3: RAD Pipeline
# ══════════════════════════════════════════════════════════════════
def draw_rad_pipeline():
    fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 6)
    ax.axis('off')

    def rbox(x, y, w, h, label, fc, ec='#333', fs=8, sublabel=None):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                             facecolor=fc, edgecolor=ec, linewidth=1.2)
        ax.add_patch(box)
        hx = fc.lstrip('#')
        if len(hx) == 3:
            hx = ''.join(c*2 for c in hx)
        tc = 'white' if sum(int(hx[i:i+2], 16) for i in (0,2,4)) < 400 else '#222'
        if sublabel:
            ax.text(x+w/2, y+h/2+0.15, label, ha='center', va='center',
                    fontsize=fs, fontweight='bold', color=tc)
            ax.text(x+w/2, y+h/2-0.22, sublabel, ha='center', va='center',
                    fontsize=fs-2, color=tc, style='italic')
        else:
            ax.text(x+w/2, y+h/2, label, ha='center', va='center',
                    fontsize=fs, fontweight='bold', color=tc)

    def arrow(x1, y1, x2, y2, lbl=None, color='#333', lw=1.5, style='-'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw, ls=style))
        if lbl:
            mx, my = (x1+x2)/2, (y1+y2)/2 + 0.18
            ax.text(mx, my, lbl, ha='center', va='bottom', fontsize=6.5,
                    color=C_GRAY, style='italic')

    # ── Input ──
    rbox(0, 3.5, 1.3, 0.9, 'Test\nImage', '#888888')
    arrow(1.3, 3.95, 2.0, 3.95)

    # ── ViT Backbone ──
    rbox(2.0, 3.5, 1.5, 0.9, 'ViT\nBackbone', C_ORANGE, fs=9)
    arrow(3.5, 3.95, 4.2, 3.95, '768-d embedding')

    # ── Three parallel branches ──
    # Branch 1: Calibrated prediction (top)
    rbox(4.2, 5.0, 1.8, 0.7, 'Temperature\nCalibration', C_BLUE, fs=8)
    ax.text(5.1, 4.85, 'T = 0.566', ha='center', fontsize=6, color=C_BLUE)
    arrow(4.0, 4.2, 4.8, 5.0, color=C_BLUE)

    # Branch 2: MC Dropout (middle)
    rbox(4.2, 3.5, 1.8, 0.7, 'MC Dropout\n(T=15 passes)', C_GREEN, fs=8)
    ax.text(5.1, 3.35, 'Entropy + MI', ha='center', fontsize=6, color=C_GREEN)

    # Branch 3: FAISS retrieval (bottom)
    rbox(4.2, 2.0, 1.8, 0.7, 'FAISS\nRetrieval', C_PURPLE, fs=8)
    ax.text(5.1, 1.85, '8,241 indexed cases', ha='center', fontsize=6, color=C_PURPLE)
    arrow(4.0, 3.7, 4.8, 2.7, color=C_PURPLE)

    # ── kNN Prediction ──
    rbox(6.5, 2.0, 1.5, 0.7, 'kNN\nPrediction', C_PURPLE, fs=8)
    ax.text(7.25, 1.85, 'Top-K voting', ha='center', fontsize=6, color=C_PURPLE)
    arrow(6.0, 2.35, 6.5, 2.35)

    # ── RAD Fusion ──
    rbox(6.5, 3.5, 1.5, 0.9, 'RAD\nFusion', '#555555', fs=9)
    ax.text(7.25, 3.3, 'α=0.85 model\n+ 0.15 kNN', ha='center', fontsize=6, color='#888888')
    arrow(6.0, 5.35, 6.5, 4.2, color=C_BLUE, lbl='p_model')
    arrow(6.0, 3.85, 6.5, 3.85, lbl='uncertainty')
    arrow(7.25, 2.7, 7.25, 3.5, lbl='p_kNN')

    # ── Confidence Routing ──
    rbox(8.5, 3.3, 1.6, 1.2, 'Confidence\nRouting', '#444444', fs=9)
    arrow(8.0, 3.95, 8.5, 3.95)

    # ── Three output tiers ──
    # Auto-Report
    rbox(7.3, 0.3, 1.3, 0.7, 'Auto-Report', C_GREEN, fs=8)
    ax.text(7.95, 0.15, '76.9% @ 96.8% acc', ha='center', fontsize=5.5, color=C_GREEN)

    # Review
    rbox(8.7, 0.3, 1.0, 0.7, 'Review', '#D4A017', fs=8)
    ax.text(9.2, 0.15, '21.4%', ha='center', fontsize=5.5, color='#D4A017')

    # Escalate
    rbox(9.8, 0.3, 1.0, 0.7, 'Escalate', C_RED, fs=8)
    ax.text(10.3, 0.15, '1.7%', ha='center', fontsize=5.5, color=C_RED)

    # Arrows from routing to tiers
    arrow(8.8, 3.3, 7.95, 1.0, color=C_GREEN)
    arrow(9.3, 3.3, 9.2, 1.0, color='#D4A017')
    arrow(9.7, 3.3, 10.3, 1.0, color=C_RED)

    # ── Criteria annotations ──
    ax.text(7.95, 1.15, 'conf > 0.85\nentr < 0.5\nretrieval agrees',
            ha='center', va='bottom', fontsize=5.5, color=C_GREEN,
            bbox=dict(boxstyle='round,pad=0.2', fc=C_LTGREEN, ec=C_GREEN, alpha=0.7))

    ax.text(9.2, 1.15, 'moderate\nconfidence',
            ha='center', va='bottom', fontsize=5.5, color='#8B7500',
            bbox=dict(boxstyle='round,pad=0.2', fc='#FFF8DC', ec='#D4A017', alpha=0.7))

    ax.text(10.3, 1.15, 'conf < 0.5\nentr > 1.0',
            ha='center', va='bottom', fontsize=5.5, color=C_RED,
            bbox=dict(boxstyle='round,pad=0.2', fc=C_LTRED, ec=C_RED, alpha=0.7))

    # ── Agreement Score ──
    rbox(8.2, 5.0, 1.5, 0.7, 'Agreement\nScore', '#666666', fs=8)
    arrow(7.7, 2.7, 8.5, 5.0, color='#666', lw=1, style='--')
    arrow(6.5, 5.2, 8.2, 5.35, color='#666', lw=1, style='--')
    arrow(9.0, 5.0, 9.3, 4.5, color='#666', lw=1)

    # Title
    ax.text(5.0, 5.7, 'Retrieval-Augmented Diagnosis (RAD) Pipeline', ha='center',
            fontsize=13, fontweight='bold', color='#222')

    fig.savefig(os.path.join(OUT_DIR, 'fig_rad_pipeline.png'))
    plt.close(fig)
    print("  [3/7] fig_rad_pipeline.png")


# ══════════════════════════════════════════════════════════════════
# FIGURE 4: Class Distribution
# ══════════════════════════════════════════════════════════════════
def draw_class_distribution():
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))

    classes = CLASS_NAMES
    train = [3075, 4489, 272, 220, 185]
    calib = [689, 982, 58, 47, 40]
    test  = [484, 837, 58, 48, 40]
    total = [t+c+te for t, c, te in zip(train, calib, test)]

    x = np.arange(len(classes))
    w = 0.25

    bars_train = ax.bar(x - w, train, w, label='Train (71.5%)', color=C_BLUE, edgecolor='white')
    bars_calib = ax.bar(x, calib, w, label='Calib (15.8%)', color=C_GREEN, edgecolor='white')
    bars_test  = ax.bar(x + w, test, w, label='Test (12.7%)', color=C_ORANGE, edgecolor='white')

    # Add total labels on top
    for i, t in enumerate(total):
        ax.text(i, max(train[i], calib[i], test[i]) + 200, f'n={t:,}',
                ha='center', va='bottom', fontsize=8, fontweight='bold', color='#333')

    # Imbalance ratio annotation
    ax.annotate(f'21:1\nimbalance', xy=(1, 4489), xytext=(3.5, 4000),
                fontsize=9, fontweight='bold', color=C_RED, ha='center',
                arrowprops=dict(arrowstyle='->', color=C_RED, lw=1.5))

    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel('Number of Images')
    ax.set_title('Class Distribution Across Dataset Splits', fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 5500)

    fig.savefig(os.path.join(OUT_DIR, 'fig_class_distribution.png'))
    plt.close(fig)
    print("  [4/7] fig_class_distribution.png")


# ══════════════════════════════════════════════════════════════════
# FIGURE 5: Confusion Matrix Heatmap
# ══════════════════════════════════════════════════════════════════
def draw_confusion_matrix():
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    cm = np.array([
        [433, 45,  5,  1,  0],
        [ 78, 757, 0,  0,  2],
        [  8,   5, 45, 0,  0],
        [  8,   0,  0, 40, 0],
        [  3,   3,  0,  0, 34]
    ])

    # Normalize for color (row-normalized)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1, aspect='equal')

    # Add text annotations
    for i in range(5):
        for j in range(5):
            val = cm[i, j]
            pct = cm_norm[i, j]
            color = 'white' if pct > 0.5 else 'black'
            if val > 0:
                ax.text(j, i, f'{val}\n({pct:.1%})', ha='center', va='center',
                        fontsize=8, color=color, fontweight='bold' if i == j else 'normal')
            else:
                ax.text(j, i, '0', ha='center', va='center', fontsize=8, color='#999')

    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha='right')
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Test Set Confusion Matrix (n=1,467)', fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Row-Normalized Proportion', fontsize=9)

    fig.savefig(os.path.join(OUT_DIR, 'fig_confusion_matrix.png'))
    plt.close(fig)
    print("  [5/7] fig_confusion_matrix.png")


# ══════════════════════════════════════════════════════════════════
# FIGURE 6: Ablation Study Comparison
# ══════════════════════════════════════════════════════════════════
def draw_ablation():
    fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=False)

    variants = ['Base ViT', 'DANN\nonly', 'DANN +\nhard mining', 'DANN +\nmixup', 'DANN-v3\n(full)']
    acc = [85.28, 84.73, 85.89, 84.66, 89.09]
    f1  = [0.843, 0.843, 0.849, 0.821, 0.879]
    auc = [0.944, 0.937, 0.947, 0.931, 0.972]

    colors = [C_GRAY, C_LTBLUE, C_LTGREEN, C_LTORANGE, C_BLUE]
    edge_colors = ['#555', C_BLUE, C_GREEN, C_ORANGE, '#1a5276']

    x = np.arange(len(variants))

    # Accuracy
    bars = axes[0].bar(x, acc, color=colors, edgecolor=edge_colors, linewidth=1.2)
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Accuracy', fontweight='bold')
    axes[0].set_ylim(82, 91)
    for i, v in enumerate(acc):
        axes[0].text(i, v + 0.15, f'{v:.1f}%', ha='center', fontsize=7, fontweight='bold')

    # F1
    bars = axes[1].bar(x, f1, color=colors, edgecolor=edge_colors, linewidth=1.2)
    axes[1].set_ylabel('Macro F1')
    axes[1].set_title('Macro F1-Score', fontweight='bold')
    axes[1].set_ylim(0.80, 0.90)
    for i, v in enumerate(f1):
        axes[1].text(i, v + 0.001, f'{v:.3f}', ha='center', fontsize=7, fontweight='bold')

    # AUC
    bars = axes[2].bar(x, auc, color=colors, edgecolor=edge_colors, linewidth=1.2)
    axes[2].set_ylabel('Macro AUC')
    axes[2].set_title('Macro AUC-ROC', fontweight='bold')
    axes[2].set_ylim(0.92, 0.98)
    for i, v in enumerate(auc):
        axes[2].text(i, v + 0.001, f'{v:.3f}', ha='center', fontsize=7, fontweight='bold')

    for ax_i in axes:
        ax_i.set_xticks(x)
        ax_i.set_xticklabels(variants, fontsize=7)
        ax_i.spines['top'].set_visible(False)
        ax_i.spines['right'].set_visible(False)

    fig.suptitle('Ablation Study: Component Contributions', fontweight='bold', fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig_ablation.png'))
    plt.close(fig)
    print("  [6/7] fig_ablation.png")


# ══════════════════════════════════════════════════════════════════
# FIGURE 7: LODO Validation Results
# ══════════════════════════════════════════════════════════════════
def draw_lodo():
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    domains = ['APTOS', 'MESSIDOR-2', 'ODIR', 'REFUGE2']
    acc = [70.8, 61.6, 51.8, 88.8]
    wf1 = [0.829, 0.633, 0.439, 0.904]
    classes_info = ['DR only', 'Nor+DR', 'All 5', 'Nor+Gla']
    n_images = [3662, 1744, 4878, 1240]
    colors = [C_BLUE, C_ORANGE, C_GREEN, C_PURPLE]

    x = np.arange(len(domains))

    # Accuracy
    bars = axes[0].bar(x, acc, color=colors, edgecolor='#333', linewidth=1)
    axes[0].axhline(y=68.2, color=C_RED, linestyle='--', linewidth=1.2, label='Average (68.2%)')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('LODO Accuracy', fontweight='bold')
    axes[0].set_ylim(0, 100)
    axes[0].legend(fontsize=8)
    for i, (v, ci) in enumerate(zip(acc, classes_info)):
        axes[0].text(i, v + 1.5, f'{v:.1f}%', ha='center', fontsize=8, fontweight='bold')
        axes[0].text(i, v - 5, ci, ha='center', fontsize=6.5, color='white', fontweight='bold')

    # Weighted F1
    bars = axes[1].bar(x, wf1, color=colors, edgecolor='#333', linewidth=1)
    axes[1].axhline(y=0.701, color=C_RED, linestyle='--', linewidth=1.2, label='Average (0.701)')
    axes[1].set_ylabel('Weighted F1-Score')
    axes[1].set_title('LODO Weighted F1', fontweight='bold')
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(fontsize=8)
    for i, v in enumerate(wf1):
        axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=8, fontweight='bold')

    for ax_i in axes:
        ax_i.set_xticks(x)
        ax_i.set_xticklabels(domains, fontsize=8)
        ax_i.spines['top'].set_visible(False)
        ax_i.spines['right'].set_visible(False)

    fig.suptitle('Leave-One-Domain-Out (LODO) Validation', fontweight='bold', fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig_lodo.png'))
    plt.close(fig)
    print("  [7/7] fig_lodo.png")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("Generating RetinaSense IEEE paper figures...")
    print(f"Output directory: {OUT_DIR}\n")

    draw_system_architecture()
    draw_dann_architecture()
    draw_rad_pipeline()
    draw_class_distribution()
    draw_confusion_matrix()
    draw_ablation()
    draw_lodo()

    print(f"\nDone! {len(os.listdir(OUT_DIR))} figures saved to {OUT_DIR}/")
