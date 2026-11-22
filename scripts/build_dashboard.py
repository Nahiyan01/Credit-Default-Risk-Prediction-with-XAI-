"""
Script to generate the comprehensive Credit Risk AI Mastery Dashboard.
Encodes figures to base64, embeds empirical metrics, builds interactive
underwriting simulator, SHAP waterfall visualizer, code explorer, and viva flashcards.
"""

import os
import base64
import json
import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = BASE_DIR / "reports" / "figures"
REPORTS_DIR = BASE_DIR / "reports"
OUTPUT_PATH = BASE_DIR / "dashboard.html"
REPORT_OUTPUT_PATH = REPORTS_DIR / "Credit_Risk_Mastery_Dashboard.html"

def get_base64_img(filename):
    path = FIGURES_DIR / filename
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

def build_dashboard():
    # Load figures
    img_roc_pr = get_base64_img("roc_pr_curves.png")
    img_cal = get_base64_img("calibration_curves.png")
    img_model_comp = get_base64_img("model_comparison.png")
    img_shap_beeswarm = get_base64_img("shap_beeswarm.png")
    img_shap_summary = get_base64_img("shap_summary_bar.png")
    img_shap_declined = get_base64_img("shap_waterfall_applicant_rejected.png")
    img_shap_approved = get_base64_img("shap_waterfall_applicant_approved.png")
    img_shap_uncertain = get_base64_img("shap_waterfall_applicant_uncertain.png")
    img_shap_stability = get_base64_img("shap_cross_val_stability.png")
    img_uncertainty = get_base64_img("uncertainty_distribution.png")

    template = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Credit Risk AI & SHAP Mastery Dashboard</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    body {
      font-family: 'Plus Jakarta Sans', sans-serif;
      background-color: #0b0f19;
      color: #e2e8f0;
    }
    code, pre, .font-mono {
      font-family: 'JetBrains Mono', monospace;
    }
    .glass-card {
      background: rgba(17, 24, 39, 0.75);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .glass-card-hover:hover {
      border-color: rgba(99, 102, 241, 0.4);
      box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.15);
    }
    .tab-btn.active {
      background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%);
      color: white;
      box-shadow: 0 4px 14px 0 rgba(79, 70, 229, 0.35);
    }
    ::-webkit-scrollbar {
      width: 8px;
      height: 8px;
    }
    ::-webkit-scrollbar-track {
      background: #0f172a;
    }
    ::-webkit-scrollbar-thumb {
      background: #334155;
      border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
      background: #475569;
    }
  </style>
</head>
<body class="min-h-screen flex flex-col antialiased selection:bg-indigo-500 selection:text-white">

  <!-- Top Announcement Bar -->
  <div class="bg-gradient-to-r from-indigo-900/60 via-purple-900/60 to-blue-900/60 border-b border-indigo-500/20 px-4 py-2 text-xs text-center text-indigo-200">
    <span>🎓 <strong>University ML Lab Project</strong>: Explainable Machine Learning for Credit Default Risk Prediction (Proposal 3) &bull; Statlog German Credit Dataset &bull; Tree Ensembles &bull; Platt Scaling &bull; SHAP</span>
  </div>

  <!-- Header -->
  <header class="sticky top-0 z-50 glass-card border-b border-slate-800/80 px-6 py-4">
    <div class="max-w-7xl mx-auto flex flex-col md:flex-row md:items-center justify-between gap-4">
      <div class="flex items-center gap-3">
        <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-indigo-600 to-blue-500 flex items-center justify-center text-white shadow-lg shadow-indigo-500/25 font-bold text-xl">
          ⚡
        </div>
        <div>
          <h1 class="text-xl font-bold tracking-tight text-white flex items-center gap-2">
            Credit Risk AI & SHAP Dashboard
            <span class="text-xs px-2.5 py-0.5 rounded-full bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 font-medium">Underwriting & XAI</span>
          </h1>
          <p class="text-xs text-slate-400">Complete Educational & Operational Guide &bull; From Theory & Math to Code & Regulatory Underwriting</p>
        </div>
      </div>

      <!-- Quick stats pill badges -->
      <div class="flex flex-wrap items-center gap-2 text-xs">
        <div class="px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700/60 flex items-center gap-1.5">
          <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
          <span class="text-slate-400">Champion:</span>
          <span class="font-semibold text-white">XGBoost (Calibrated)</span>
        </div>
        <div class="px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700/60 flex items-center gap-1.5">
          <span class="text-slate-400">ROC-AUC:</span>
          <span class="font-semibold text-indigo-400">0.8019</span>
        </div>
        <div class="px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700/60 flex items-center gap-1.5">
          <span class="text-slate-400">PR-AUC:</span>
          <span class="font-semibold text-purple-400">0.6650</span>
        </div>
        <div class="px-3 py-1.5 rounded-lg bg-slate-800/80 border border-slate-700/60 flex items-center gap-1.5">
          <span class="text-slate-400">Brier Imp:</span>
          <span class="font-semibold text-emerald-400">+9.7%</span>
        </div>
      </div>
    </div>
  </header>

  <!-- Navigation Tabs -->
  <div class="bg-slate-900/90 border-b border-slate-800 px-6 py-2 sticky top-[73px] z-40">
    <div class="max-w-7xl mx-auto flex items-center gap-2 overflow-x-auto no-scrollbar py-1">
      <button onclick="switchTab('overview')" id="tab-btn-overview" class="tab-btn active px-4 py-2 rounded-lg text-sm font-medium transition-all flex items-center gap-2 whitespace-nowrap">
        <span>🌐</span> Project Overview & Story
      </button>
      <button onclick="switchTab('simulator')" id="tab-btn-simulator" class="tab-btn px-4 py-2 rounded-lg text-sm font-medium text-slate-400 hover:text-white hover:bg-slate-800 transition-all flex items-center gap-2 whitespace-nowrap">
        <span>🧪</span> Live Underwriting Simulator
      </button>
      <button onclick="switchTab('theory')" id="tab-btn-theory" class="tab-btn px-4 py-2 rounded-lg text-sm font-medium text-slate-400 hover:text-white hover:bg-slate-800 transition-all flex items-center gap-2 whitespace-nowrap">
        <span>🧠</span> ML Theory (Zero-Jargon)
      </button>
      <button onclick="switchTab('benchmark')" id="tab-btn-benchmark" class="tab-btn px-4 py-2 rounded-lg text-sm font-medium text-slate-400 hover:text-white hover:bg-slate-800 transition-all flex items-center gap-2 whitespace-nowrap">
        <span>📊</span> Model Benchmarks & Metrics
      </button>
      <button onclick="switchTab('shap')" id="tab-btn-shap" class="tab-btn px-4 py-2 rounded-lg text-sm font-medium text-slate-400 hover:text-white hover:bg-slate-800 transition-all flex items-center gap-2 whitespace-nowrap">
        <span>🔍</span> SHAP Explainability & Cases
      </button>
      <button onclick="switchTab('code')" id="tab-btn-code" class="tab-btn px-4 py-2 rounded-lg text-sm font-medium text-slate-400 hover:text-white hover:bg-slate-800 transition-all flex items-center gap-2 whitespace-nowrap">
        <span>💻</span> Code Architecture Walkthrough
      </button>
      <button onclick="switchTab('viva')" id="tab-btn-viva" class="tab-btn px-4 py-2 rounded-lg text-sm font-medium text-slate-400 hover:text-white hover:bg-slate-800 transition-all flex items-center gap-2 whitespace-nowrap">
        <span>🎓</span> Viva / Exam Defense Prep
      </button>
    </div>
  </div>

  <!-- Main Content Container -->
  <main class="flex-1 max-w-7xl w-full mx-auto p-6 space-y-8">

    <!-- TAB 1: OVERVIEW & STORY -->
    <div id="tab-content-overview" class="space-y-8">
      
      <div class="glass-card rounded-2xl p-8 border border-indigo-500/20 relative overflow-hidden bg-gradient-to-br from-indigo-950/40 via-slate-900 to-slate-950">
        <div class="absolute -right-10 -bottom-10 w-96 h-96 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none"></div>
        <div class="relative z-10 max-w-3xl space-y-4">
          <span class="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
            <span>💡</span> The Intuitive Big Picture
          </span>
          <h2 class="text-3xl font-extrabold text-white tracking-tight leading-tight">
            How Do Banks Use AI to Decide Loans Without Breaking the Law?
          </h2>
          <p class="text-slate-300 text-sm leading-relaxed">
            Every day, banks receive thousands of loan applications. If they lend to someone who defaults (doesn't pay back), the bank loses money. But if they use a "black-box" AI that just says <span class="text-amber-300 font-mono">"Denied"</span> without reasons, they violate federal consumer financial protection laws (ECOA & FCRA).
          </p>
          <div class="pt-2 flex flex-wrap gap-4 text-xs">
            <div class="flex items-center gap-2 text-slate-400">
              <span class="text-emerald-400 font-bold text-base">✓</span> <strong>Predictive Accuracy</strong> with Tuned Tree Ensembles
            </div>
            <div class="flex items-center gap-2 text-slate-400">
              <span class="text-indigo-400 font-bold text-base">✓</span> <strong>Honest Probabilities</strong> with Platt Scaling Calibration
            </div>
            <div class="flex items-center gap-2 text-slate-400">
              <span class="text-purple-400 font-bold text-base">✓</span> <strong>Legally Auditable Reasons</strong> with SHAP Game Theory
            </div>
          </div>
        </div>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div class="glass-card rounded-xl p-6 space-y-3 border-t-4 border-t-blue-500">
          <div class="w-10 h-10 rounded-lg bg-blue-500/10 text-blue-400 flex items-center justify-center font-bold text-lg">1</div>
          <h3 class="text-base font-bold text-white">The Bank's Dilemma</h3>
          <p class="text-xs text-slate-400 leading-relaxed">
            Out of 1,000 borrowers in our dataset, 700 repay on time and 300 default. Blindly rejecting people destroys customer loyalty, while approving bad loans leads to balance sheet write-offs. We need an intelligent, automated risk engine.
          </p>
        </div>

        <div class="glass-card rounded-xl p-6 space-y-3 border-t-4 border-t-purple-500">
          <div class="w-10 h-10 rounded-lg bg-purple-500/10 text-purple-400 flex items-center justify-center font-bold text-lg">2</div>
          <h3 class="text-base font-bold text-white">The Legal Mandate (FCRA & ECOA)</h3>
          <p class="text-xs text-slate-400 leading-relaxed">
            Under ECOA (Regulation B) and FCRA § 615(a), banks are legally prohibited from discriminating and <strong>must issue an Adverse Action Notice</strong> giving the top 4 specific financial factors that caused a loan denial.
          </p>
        </div>

        <div class="glass-card rounded-xl p-6 space-y-3 border-t-4 border-t-emerald-500">
          <div class="w-10 h-10 rounded-lg bg-emerald-500/10 text-emerald-400 flex items-center justify-center font-bold text-lg">3</div>
          <h3 class="text-base font-bold text-white">Our 3-Tier Underwriting Protocol</h3>
          <p class="text-xs text-slate-400 leading-relaxed">
            Instead of binary yes/no, our system implements a smart 3-tier triage:
            <br>&bull; <strong class="text-emerald-400">P &lt; 30%</strong>: Auto-Approve (Fast track)
            <br>&bull; <strong class="text-amber-400">30% &le; P &le; 65%</strong>: Human Underwriter Review
            <br>&bull; <strong class="text-rose-400">P &gt; 65%</strong>: Auto-Decline + SHAP Adverse Reasons
          </p>
        </div>
      </div>

      <div class="glass-card rounded-xl p-6 space-y-4">
        <h3 class="text-sm font-bold text-white uppercase tracking-wider text-slate-400 flex items-center gap-2">
          <span>⚙️</span> End-to-End System Architecture
        </h3>
        <div class="grid grid-cols-1 md:grid-cols-5 gap-3 text-center">
          <div class="p-4 rounded-xl bg-slate-800/60 border border-slate-700/50 space-y-2">
            <div class="text-2xl">📥</div>
            <div class="text-xs font-bold text-white">1. Loan Application</div>
            <div class="text-[11px] text-slate-400">20 Financial & Demographic Features</div>
          </div>
          <div class="p-4 rounded-xl bg-slate-800/60 border border-slate-700/50 space-y-2">
            <div class="text-2xl">🧹</div>
            <div class="text-xs font-bold text-white">2. Preprocessing</div>
            <div class="text-[11px] text-slate-400">StandardScaler + OneHotEncoder (Leakage-Free)</div>
          </div>
          <div class="p-4 rounded-xl bg-slate-800/60 border border-indigo-500/40 bg-indigo-950/20 space-y-2">
            <div class="text-2xl">🌲</div>
            <div class="text-xs font-bold text-indigo-300">3. XGBoost Model</div>
            <div class="text-[11px] text-slate-400">Tuned with Stratified 5-Fold Cross-Validation</div>
          </div>
          <div class="p-4 rounded-xl bg-slate-800/60 border border-purple-500/40 bg-purple-950/20 space-y-2">
            <div class="text-2xl">📈</div>
            <div class="text-xs font-bold text-purple-300">4. Platt Calibration</div>
            <div class="text-[11px] text-slate-400">Sigmoid transform produces honest P(Default)</div>
          </div>
          <div class="p-4 rounded-xl bg-slate-800/60 border border-emerald-500/40 bg-emerald-950/20 space-y-2">
            <div class="text-2xl">📑</div>
            <div class="text-xs font-bold text-emerald-300">5. SHAP Explanations</div>
            <div class="text-[11px] text-slate-400">3-Tier Decision & Adverse Action Notice</div>
          </div>
        </div>
      </div>

    </div>

    <!-- TAB 2: LIVE UNDERWRITING SIMULATOR -->
    <div id="tab-content-simulator" class="hidden space-y-8">
      
      <div class="glass-card rounded-2xl p-6 border border-emerald-500/30 bg-gradient-to-r from-emerald-950/30 via-slate-900 to-slate-950 flex flex-col md:flex-row items-center justify-between gap-6">
        <div class="space-y-2">
          <span class="inline-flex items-center gap-1.5 px-3 py-0.5 rounded-full text-xs font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
            <span>🎮</span> Live Interactive Underwriting Portal
          </span>
          <h2 class="text-2xl font-bold text-white">Test Real-Time Loan Applications & SHAP Attribution</h2>
          <p class="text-xs text-slate-400 max-w-2xl">
            Move the borrower sliders below. Watch how the calibrated probability of default shifts, triggers the 3-Tier underwriting policy, decomposes into SHAP risk drivers, and automatically drafts an FCRA-compliant Adverse Action notice!
          </p>
        </div>
        <button onclick="resetSimulator()" class="px-4 py-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs font-medium text-slate-300 border border-slate-700 transition-all flex items-center gap-1.5 whitespace-nowrap">
          <span>🔄</span> Reset to Applicant #37
        </button>
      </div>

      <div class="grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        <div class="lg:col-span-7 glass-card rounded-2xl p-6 space-y-6">
          <h3 class="text-sm font-bold text-white uppercase tracking-wider text-slate-400 flex items-center justify-between">
            <span>Borrower Financial Profile</span>
            <span class="text-xs normal-case text-indigo-400 font-normal">Auto-recalculating on change</span>
          </h3>

          <div class="space-y-4">
            
            <div class="space-y-1.5">
              <div class="flex justify-between text-xs">
                <label class="font-medium text-slate-300">Loan Duration (Months)</label>
                <span id="val-duration" class="font-mono text-indigo-400 font-bold">48 months</span>
              </div>
              <input type="range" id="input-duration" min="4" max="72" value="48" step="1" oninput="updateSimulator()" class="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500">
              <div class="flex justify-between text-[10px] text-slate-500">
                <span>4 mos (Short, Safe)</span>
                <span>24 mos (Typical)</span>
                <span>72 mos (Long, High Risk)</span>
              </div>
            </div>

            <div class="space-y-1.5">
              <div class="flex justify-between text-xs">
                <label class="font-medium text-slate-300">Requested Credit Amount (DM / Currency)</label>
                <span id="val-amount" class="font-mono text-indigo-400 font-bold">3,915 DM</span>
              </div>
              <input type="range" id="input-amount" min="500" max="15000" value="3915" step="250" oninput="updateSimulator()" class="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500">
              <div class="flex justify-between text-[10px] text-slate-500">
                <span>500 DM (Micro)</span>
                <span>4,000 DM (Medium)</span>
                <span>15,000 DM (Heavy Debt)</span>
              </div>
            </div>

            <div class="space-y-1.5">
              <label class="block text-xs font-medium text-slate-300">Checking Account Balance Status</label>
              <select id="input-checking" onchange="updateSimulator()" class="w-full px-3 py-2 rounded-lg bg-slate-800/90 border border-slate-700 text-xs text-white focus:outline-none focus:border-indigo-500">
                <option value="lt_0" selected>&lt; 0 DM (Overdrawn / Negative Balance) [Highest Risk]</option>
                <option value="0_to_200">0 &le; Balance &lt; 200 DM (Modest Liquidity)</option>
                <option value="ge_200">&ge; 200 DM (Healthy Buffer)</option>
                <option value="no_checking">No Checking Account Registered (Neutral/Established)</option>
              </select>
            </div>

            <div class="space-y-1.5">
              <label class="block text-xs font-medium text-slate-300">Savings / Investment Reserves</label>
              <select id="input-savings" onchange="updateSimulator()" class="w-full px-3 py-2 rounded-lg bg-slate-800/90 border border-slate-700 text-xs text-white focus:outline-none focus:border-indigo-500">
                <option value="lt_100" selected>&lt; 100 DM (Minimal Reserves)</option>
                <option value="100_to_500">100 &le; Balance &lt; 500 DM</option>
                <option value="500_to_1000">500 &le; Balance &lt; 1,000 DM</option>
                <option value="ge_1000">&ge; 1,000 DM (Strong Reserves)</option>
                <option value="unknown">Unknown / No Records</option>
              </select>
            </div>

            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div class="space-y-1.5">
                <div class="flex justify-between text-xs">
                  <label class="font-medium text-slate-300">Applicant Age</label>
                  <span id="val-age" class="font-mono text-indigo-400 font-bold">25 yrs</span>
                </div>
                <input type="range" id="input-age" min="19" max="75" value="25" step="1" oninput="updateSimulator()" class="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-500">
              </div>

              <div class="space-y-1.5">
                <label class="block text-xs font-medium text-slate-300">Housing Status</label>
                <select id="input-housing" onchange="updateSimulator()" class="w-full px-3 py-2 rounded-lg bg-slate-800/90 border border-slate-700 text-xs text-white focus:outline-none focus:border-indigo-500">
                  <option value="own">Owns Home (Lower Risk)</option>
                  <option value="rent" selected>Rents Apartment</option>
                  <option value="for_free">Living For Free (Parents/Employer)</option>
                </select>
              </div>
            </div>

            <div class="space-y-1.5">
              <label class="block text-xs font-medium text-slate-300">Current Employment Stability</label>
              <select id="input-employment" onchange="updateSimulator()" class="w-full px-3 py-2 rounded-lg bg-slate-800/90 border border-slate-700 text-xs text-white focus:outline-none focus:border-indigo-500">
                <option value="unemployed">Unemployed</option>
                <option value="lt_1" selected>&lt; 1 year on job</option>
                <option value="1_to_4">1 to 4 years</option>
                <option value="4_to_7">4 to 7 years</option>
                <option value="ge_7">&ge; 7 years (Tenured)</option>
              </select>
            </div>

          </div>
        </div>

        <div class="lg:col-span-5 space-y-6">
          
          <div id="sim-decision-card" class="glass-card rounded-2xl p-6 border border-rose-500/40 bg-gradient-to-br from-rose-950/20 to-slate-900 space-y-4">
            <div class="flex items-center justify-between">
              <span class="text-xs text-slate-400 uppercase tracking-wider font-semibold">Underwriting Disposition</span>
              <span id="sim-tier-badge" class="px-3 py-1 rounded-full text-xs font-bold bg-rose-500/20 text-rose-300 border border-rose-500/40">
                TIER 3: AUTO-DECLINE
              </span>
            </div>

            <div class="space-y-2">
              <div class="flex justify-between items-baseline">
                <span class="text-xs text-slate-400">Calibrated Default Risk P(Default)</span>
                <span id="sim-prob-text" class="text-3xl font-extrabold text-rose-400 font-mono">74.5%</span>
              </div>
              <div class="w-full bg-slate-800 h-3 rounded-full overflow-hidden flex">
                <div id="sim-prob-bar" class="bg-gradient-to-r from-amber-500 to-rose-500 h-full transition-all duration-300" style="width: 74.5%;"></div>
              </div>
              <div class="flex justify-between text-[10px] text-slate-500 font-mono">
                <span>0% Safe</span>
                <span class="text-emerald-400 font-bold">30% (Auto-Approve Cutoff)</span>
                <span class="text-rose-400 font-bold">65% (Decline Cutoff)</span>
                <span>100%</span>
              </div>
            </div>

            <div id="sim-action-text" class="p-3 rounded-lg bg-slate-950/60 border border-slate-800 text-xs text-slate-300 leading-relaxed">
              Default risk exceeds the 65% automatic decline threshold. Application is rejected immediately, and an FCRA Adverse Action Statement must be provided.
            </div>
          </div>

          <div class="glass-card rounded-2xl p-6 space-y-4">
            <h4 class="text-xs font-bold text-white uppercase tracking-wider text-slate-400 flex items-center justify-between">
              <span>SHAP Feature Attributions</span>
              <span class="text-[10px] normal-case text-slate-400 font-mono">Base log-odds E[f(x)] = -0.85</span>
            </h4>
            <p class="text-[11px] text-slate-400">
              Red bars increase default risk; Green bars mitigate risk.
            </p>

            <div id="sim-shap-bars" class="space-y-2 text-xs">
            </div>
          </div>

          <div id="sim-adverse-box" class="glass-card rounded-2xl p-5 border border-amber-500/30 bg-amber-950/10 space-y-3">
            <div class="flex items-center gap-2 text-xs font-bold text-amber-400">
              <span>📋</span> Official FCRA § 615(a) Adverse Action Reasons
            </div>
            <ul id="sim-adverse-list" class="space-y-1.5 text-xs text-slate-300 list-disc list-inside">
            </ul>
          </div>

        </div>

      </div>

    </div>

    <!-- TAB 3: MACHINE LEARNING THEORY -->
    <div id="tab-content-theory" class="hidden space-y-8">
      
      <div class="glass-card rounded-2xl p-8 border border-purple-500/20 bg-gradient-to-r from-purple-950/20 via-slate-900 to-slate-950 space-y-4">
        <span class="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-purple-500/20 text-purple-300 border border-purple-500/30">
          <span>📚</span> From First Principles to Advanced Math
        </span>
        <h2 class="text-2xl font-bold text-white">Machine Learning Theory Explained with Everyday Analogies</h2>
        <p class="text-xs text-slate-300 max-w-3xl leading-relaxed">
          If you missed classes, don't worry. Machine learning is just teaching a computer to recognize patterns from past experience. Here are the core concepts that power this project.
        </p>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        
        <div class="glass-card rounded-xl p-6 space-y-4">
          <div class="flex items-center gap-3">
            <span class="text-2xl">⚖️</span>
            <div>
              <h3 class="text-base font-bold text-white">1. The Class Imbalance Trap</h3>
              <span class="text-xs text-slate-400">Why simple accuracy will ruin your bank</span>
            </div>
          </div>
          <div class="p-3 rounded-lg bg-slate-950/60 border border-slate-800 text-xs text-slate-300 space-y-2">
            <p>Our dataset has <strong>700 good borrowers (70%)</strong> and <strong>300 bad borrowers (30%)</strong>.</p>
            <p>If a lazy AI just says <em>"Everyone is good"</em> for all 1,000 people, it gets <strong>70% accuracy</strong>! The teacher would think it's decent, but the bank would lose millions on all 300 defaulters.</p>
          </div>
          <div class="text-xs space-y-2 text-slate-400">
            <div class="font-semibold text-slate-300">How we solve it in this project:</div>
            <div>&bull; <strong class="text-indigo-400">Class-Weighted Loss</strong>: We multiply the loss penalty on minority default errors by w1 = N_0 / N_1 ≈ 2.33. The AI gets punished 2.33x harder if it misses a default.</div>
            <div>&bull; <strong class="text-purple-400">SMOTE</strong>: Creates synthetic, realistic artificial default examples along lines connecting neighboring default records.</div>
          </div>
        </div>

        <div class="glass-card rounded-xl p-6 space-y-4">
          <div class="flex items-center gap-3">
            <span class="text-2xl">🌲</span>
            <div>
              <h3 class="text-base font-bold text-white">2. Logistic Regression vs Tree Ensembles</h3>
              <span class="text-xs text-slate-400">Linear models vs Gradient Boosted Trees</span>
            </div>
          </div>
          <div class="p-3 rounded-lg bg-slate-950/60 border border-slate-800 text-xs text-slate-300 space-y-2">
            <p><strong>Logistic Regression</strong>: Draws a smooth single hyperplane separating good and bad applicants. Great baseline, but cannot easily capture complex non-linear combinations (e.g. young age + high debt).</p>
            <p><strong>Random Forest</strong>: A "parliament" of 200 independent trees voting. Reduces variance (overfitting).</p>
            <p><strong>XGBoost / LightGBM</strong>: A team where each new tree specifically trains on the residual errors of the previous trees. This won the highest PR-AUC (0.6650).</p>
          </div>
        </div>

        <div class="glass-card rounded-xl p-6 space-y-4">
          <div class="flex items-center gap-3">
            <span class="text-2xl">📈</span>
            <div>
              <h3 class="text-base font-bold text-white">3. Platt Scaling (Probability Calibration)</h3>
              <span class="text-xs text-slate-400">Turning raw scores into honest probabilities</span>
            </div>
          </div>
          <div class="p-3 rounded-lg bg-slate-950/60 border border-slate-800 text-xs text-slate-300 space-y-2">
            <p>XGBoost outputs a margin f(x). If the model is uncalibrated, a raw score of 0.85 might only mean a 60% real-world default rate!</p>
            <p class="font-mono text-indigo-300 text-[11px]">P(Y=1 | f(x)) = 1 / (1 + exp(A &bull; f(x) + B))</p>
            <p>Platt scaling fits a post-hoc logistic transformation over cross-validation holdouts. This reduced our <strong>Brier Score</strong> from <strong>0.1740 &rarr; 0.1570</strong> (a 9.7% improvement in risk honesty).</p>
          </div>
        </div>

        <div class="glass-card rounded-xl p-6 space-y-4">
          <div class="flex items-center gap-3">
            <span class="text-2xl">🔍</span>
            <div>
              <h3 class="text-base font-bold text-white">4. SHAP (Shapley Additive exPlanations)</h3>
              <span class="text-xs text-slate-400">Nobel Prize economics applied to AI transparency</span>
            </div>
          </div>
          <div class="p-3 rounded-lg bg-slate-950/60 border border-slate-800 text-xs text-slate-300 space-y-2">
            <p>Imagine a soccer team wins 3-1. How much credit goes to the striker vs goalkeeper? Lloyd Shapley invented <strong>Shapley Values</strong> (1953) to fairly share credit in cooperative games.</p>
            <p>In our model, the "players" are features (Duration, Checking Status, Age). SHAP proves mathematically that:</p>
            <p class="font-mono text-purple-300 text-[11px]">f(x) = E[f(x)] + &sum; &phi;<sub>i</sub>(x)</p>
            <p>The total prediction equals the base risk plus each feature's marginal contribution!</p>
          </div>
        </div>

      </div>

    </div>

    <!-- TAB 4: BENCHMARK & METRICS -->
    <div id="tab-content-benchmark" class="hidden space-y-8">
      
      <div class="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 class="text-xl font-bold text-white">Empirical Model Benchmarks</h2>
          <p class="text-xs text-slate-400">Evaluated on the 200-applicant holdout test set (Stratified 20% split)</p>
        </div>
        <div class="text-xs text-indigo-400 bg-indigo-500/10 border border-indigo-500/20 px-3 py-1.5 rounded-lg">
          Champion: <strong>XGBoost (Calibrated)</strong> &bull; Highest PR-AUC (0.6650)
        </div>
      </div>

      <div class="glass-card rounded-xl overflow-hidden border border-slate-800">
        <div class="overflow-x-auto">
          <table class="w-full text-left text-xs text-slate-300">
            <thead class="bg-slate-900/90 text-slate-400 uppercase font-semibold text-[11px] border-b border-slate-800">
              <tr>
                <th class="py-3 px-4">Model Architecture</th>
                <th class="py-3 px-4">ROC-AUC</th>
                <th class="py-3 px-4">PR-AUC (Default Class)</th>
                <th class="py-3 px-4">F1-Score</th>
                <th class="py-3 px-4">FPR (Denying Good)</th>
                <th class="py-3 px-4">Brier Score</th>
                <th class="py-3 px-4">Champion Status</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-slate-800 font-mono">
              <tr class="bg-indigo-950/20 text-white font-medium">
                <td class="py-3.5 px-4 font-sans font-bold flex items-center gap-2">
                  <span class="w-2 h-2 rounded-full bg-emerald-400"></span> XGBoost (Calibrated)
                </td>
                <td class="py-3.5 px-4 text-indigo-300 font-bold">0.8019</td>
                <td class="py-3.5 px-4 text-purple-300 font-bold">0.6650</td>
                <td class="py-3.5 px-4">0.5400</td>
                <td class="py-3.5 px-4 text-emerald-400">9.29%</td>
                <td class="py-3.5 px-4 text-emerald-400">0.1570</td>
                <td class="py-3.5 px-4 font-sans"><span class="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 text-[10px] font-bold">CHAMPION</span></td>
              </tr>
              <tr class="hover:bg-slate-800/40">
                <td class="py-3 px-4 font-sans font-medium text-slate-200">Logistic Regression (Calibrated)</td>
                <td class="py-3 px-4 text-slate-300">0.8101</td>
                <td class="py-3 px-4 text-slate-300">0.6475</td>
                <td class="py-3 px-4">0.5882</td>
                <td class="py-3 px-4">8.57%</td>
                <td class="py-3 px-4">0.1546</td>
                <td class="py-3 px-4 font-sans text-slate-500 text-[11px]">Strong Linear Baseline</td>
              </tr>
              <tr class="hover:bg-slate-800/40">
                <td class="py-3 px-4 font-sans font-medium text-slate-200">XGBoost + SMOTE (Calibrated)</td>
                <td class="py-3 px-4 text-slate-300">0.8023</td>
                <td class="py-3 px-4 text-slate-300">0.6510</td>
                <td class="py-3 px-4">0.5361</td>
                <td class="py-3 px-4">7.86%</td>
                <td class="py-3 px-4">0.1574</td>
                <td class="py-3 px-4 font-sans text-slate-500 text-[11px]">Resampled Variant</td>
              </tr>
              <tr class="hover:bg-slate-800/40">
                <td class="py-3 px-4 font-sans font-medium text-slate-200">Random Forest (Calibrated)</td>
                <td class="py-3 px-4 text-slate-300">0.7951</td>
                <td class="py-3 px-4 text-slate-300">0.6390</td>
                <td class="py-3 px-4">0.5472</td>
                <td class="py-3 px-4">12.14%</td>
                <td class="py-3 px-4">0.1605</td>
                <td class="py-3 px-4 font-sans text-slate-500 text-[11px]">Bagging Baseline</td>
              </tr>
              <tr class="hover:bg-slate-800/40">
                <td class="py-3 px-4 font-sans font-medium text-slate-200">LightGBM (Calibrated)</td>
                <td class="py-3 px-4 text-slate-300">0.7830</td>
                <td class="py-3 px-4 text-slate-300">0.6517</td>
                <td class="py-3 px-4">0.5051</td>
                <td class="py-3 px-4">10.00%</td>
                <td class="py-3 px-4">0.1637</td>
                <td class="py-3 px-4 font-sans text-slate-500 text-[11px]">Leaf-Wise Boosting</td>
              </tr>
              <tr class="hover:bg-slate-800/40 text-slate-400">
                <td class="py-3 px-4 font-sans font-medium">XGBoost (Uncalibrated)</td>
                <td class="py-3 px-4">0.7932</td>
                <td class="py-3 px-4">0.6450</td>
                <td class="py-3 px-4">0.6202</td>
                <td class="py-3 px-4 text-rose-400">20.71%</td>
                <td class="py-3 px-4 text-rose-400">0.1740</td>
                <td class="py-3 px-4 font-sans text-rose-400 text-[11px]">High False Rejections</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div class="glass-card rounded-xl p-5 space-y-3">
          <div class="flex items-center justify-between">
            <h3 class="text-sm font-bold text-white">ROC Curves & Precision-Recall Curves</h3>
            <span class="text-[10px] text-slate-400 bg-slate-800 px-2 py-0.5 rounded">All 5 Models</span>
          </div>
          <p class="text-xs text-slate-400">Notice PR-AUC on the right: XGBoost leads with 0.665, capturing the rare 30% defaulter class with superior precision.</p>
          <div class="rounded-lg overflow-hidden border border-slate-800 bg-slate-950">
            <img src="__IMG_ROC_PR__" alt="ROC and PR Curves" class="w-full h-auto object-contain hover:scale-105 transition-transform duration-300">
          </div>
        </div>

        <div class="glass-card rounded-xl p-5 space-y-3">
          <div class="flex items-center justify-between">
            <h3 class="text-sm font-bold text-white">Calibration Reliability Curves</h3>
            <span class="text-[10px] text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-2 py-0.5 rounded">Platt Scaling</span>
          </div>
          <p class="text-xs text-slate-400">The diagonal dotted line represents perfect calibration. Platt-scaled XGBoost (green) aligns tightly with true default frequencies.</p>
          <div class="rounded-lg overflow-hidden border border-slate-800 bg-slate-950">
            <img src="__IMG_CAL__" alt="Calibration Curves" class="w-full h-auto object-contain hover:scale-105 transition-transform duration-300">
          </div>
        </div>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div class="glass-card rounded-xl p-5 space-y-3">
          <div class="flex items-center justify-between">
            <h3 class="text-sm font-bold text-white">Model Metrics Comparison Bar Chart</h3>
            <span class="text-[10px] text-slate-400 bg-slate-800 px-2 py-0.5 rounded">Multi-Metric</span>
          </div>
          <p class="text-xs text-slate-400">Comparing ROC-AUC, PR-AUC, F1, and False Positive Rate side-by-side.</p>
          <div class="rounded-lg overflow-hidden border border-slate-800 bg-slate-950">
            <img src="__IMG_MODEL_COMP__" alt="Model Comparison" class="w-full h-auto object-contain hover:scale-105 transition-transform duration-300">
          </div>
        </div>

        <div class="glass-card rounded-xl p-5 space-y-3">
          <div class="flex items-center justify-between">
            <h3 class="text-sm font-bold text-white">Model Uncertainty & Triage Distribution</h3>
            <span class="text-[10px] text-indigo-400 bg-indigo-500/10 border border-indigo-500/20 px-2 py-0.5 rounded">3-Tier Policy</span>
          </div>
          <p class="text-xs text-slate-400">How the test partition splits across Tier 1 (Auto-Approve), Tier 2 (Human Review), and Tier 3 (Auto-Decline).</p>
          <div class="rounded-lg overflow-hidden border border-slate-800 bg-slate-950">
            <img src="__IMG_UNCERTAINTY_DIST__" alt="Uncertainty Distribution" class="w-full h-auto object-contain hover:scale-105 transition-transform duration-300">
          </div>
        </div>
      </div>

    </div>

    <!-- TAB 5: SHAP EXPLAINABILITY -->
    <div id="tab-content-shap" class="hidden space-y-8">
      
      <div class="glass-card rounded-2xl p-6 border border-purple-500/30 bg-gradient-to-r from-purple-950/30 via-slate-900 to-slate-950 flex flex-col md:flex-row items-center justify-between gap-6">
        <div class="space-y-2">
          <span class="inline-flex items-center gap-1.5 px-3 py-0.5 rounded-full text-xs font-semibold bg-purple-500/20 text-purple-300 border border-purple-500/30">
            <span>🔍</span> Global & Local XAI Gallery
          </span>
          <h2 class="text-2xl font-bold text-white">SHAP Attribution: Peeking Inside the Black Box</h2>
          <p class="text-xs text-slate-400 max-w-2xl">
            We use SHAP for two levels of explainability: <strong>Global Macro</strong> (which features drive risk overall across the bank) and <strong>Local Micro</strong> (waterfall breakdowns for specific individuals).
          </p>
        </div>
      </div>

      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div class="glass-card rounded-xl p-5 space-y-3">
          <div class="flex items-center justify-between">
            <h3 class="text-sm font-bold text-white">Global SHAP Beeswarm Plot</h3>
            <span class="text-[10px] text-purple-400 bg-purple-500/10 border border-purple-500/20 px-2 py-0.5 rounded">Macro Policy</span>
          </div>
          <p class="text-xs text-slate-400">Every dot is an applicant. Red = high feature value; Blue = low. High loan duration and overdrawn checking account push dots far to the right (increased default risk).</p>
          <div class="rounded-lg overflow-hidden border border-slate-800 bg-slate-950">
            <img src="__IMG_BEESWARM__" alt="SHAP Beeswarm" class="w-full h-auto object-contain hover:scale-105 transition-transform duration-300">
          </div>
        </div>

        <div class="glass-card rounded-xl p-5 space-y-3">
          <div class="flex items-center justify-between">
            <h3 class="text-sm font-bold text-white">Mean Absolute SHAP Importance Bar</h3>
            <span class="text-[10px] text-slate-400 bg-slate-800 px-2 py-0.5 rounded">Ranked Impact</span>
          </div>
          <p class="text-xs text-slate-400">Top 3 macro risk indicators across the entire portfolio: Loan Duration, Checking Account Balance Status, and Credit Amount.</p>
          <div class="rounded-lg overflow-hidden border border-slate-800 bg-slate-950">
            <img src="__IMG_SUMMARY__" alt="SHAP Summary Bar" class="w-full h-auto object-contain hover:scale-105 transition-transform duration-300">
          </div>
        </div>
      </div>

      <div class="space-y-4">
        <div class="flex items-center justify-between">
          <h3 class="text-base font-bold text-white flex items-center gap-2">
            <span>👤</span> Empirical Case Studies: 3 Underwriting Tiers
          </h3>
          <div class="flex items-center gap-2">
            <button onclick="switchCaseStudy('declined')" id="btn-case-declined" class="px-3 py-1.5 rounded-lg text-xs font-semibold bg-rose-500 text-white shadow-lg">Case 1: Auto-Declined (#37)</button>
            <button onclick="switchCaseStudy('approved')" id="btn-case-approved" class="px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-800 text-slate-300 hover:text-white">Case 2: Auto-Approved (#14)</button>
            <button onclick="switchCaseStudy('uncertain')" id="btn-case-uncertain" class="px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-800 text-slate-300 hover:text-white">Case 3: Underwriter Review (#3)</button>
          </div>
        </div>

        <div id="case-study-declined" class="glass-card rounded-xl p-6 space-y-4 border border-rose-500/30">
          <div class="flex flex-col md:flex-row md:items-center justify-between gap-2 border-b border-slate-800 pb-3">
            <div>
              <span class="text-xs text-rose-400 font-bold uppercase tracking-wider">Tier 3: Auto-Decline</span>
              <h4 class="text-lg font-bold text-white">Applicant #37 &bull; Calibrated Default Risk = 74.5%</h4>
            </div>
            <div class="text-xs text-slate-400">
              Profile: 48 month duration, 3,915 DM, Negative Checking, Savings &lt; 100 DM
            </div>
          </div>
          <div class="grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">
            <div class="lg:col-span-7 rounded-lg overflow-hidden border border-slate-800 bg-slate-950">
              <img src="__IMG_DECLINED__" alt="SHAP Waterfall Declined" class="w-full h-auto object-contain">
            </div>
            <div class="lg:col-span-5 space-y-3">
              <h5 class="text-xs font-bold text-slate-300 uppercase tracking-wider">Generated FCRA Adverse Action Disclosure</h5>
              <div class="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2 text-xs">
                <div class="text-slate-400">Top factors adversely affecting credit evaluation:</div>
                <ol class="list-decimal list-inside space-y-1 text-slate-200">
                  <li><strong>Duration</strong> (+0.795): Requested loan term is excessively long relative to risk profile.</li>
                  <li><strong>Checking Status</strong> (+0.434): Status of checking account is low, overdrawn, or absent.</li>
                  <li><strong>Credit Amount</strong> (+0.404): Requested credit amount is high relative to repayment capacity.</li>
                  <li><strong>Negative Checking</strong> (+0.230): Account currently shows overdrawn status.</li>
                </ol>
              </div>
            </div>
          </div>
        </div>

        <div id="case-study-approved" class="hidden glass-card rounded-xl p-6 space-y-4 border border-emerald-500/30">
          <div class="flex flex-col md:flex-row md:items-center justify-between gap-2 border-b border-slate-800 pb-3">
            <div>
              <span class="text-xs text-emerald-400 font-bold uppercase tracking-wider">Tier 1: Auto-Approve</span>
              <h4 class="text-lg font-bold text-white">Applicant #14 &bull; Calibrated Default Risk = 8.9%</h4>
            </div>
            <div class="text-xs text-slate-400">
              Profile: 4 month term, 1,544 DM, Established Banking, 4-7 yrs Employment
            </div>
          </div>
          <div class="grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">
            <div class="lg:col-span-7 rounded-lg overflow-hidden border border-slate-800 bg-slate-950">
              <img src="__IMG_APPROVED__" alt="SHAP Waterfall Approved" class="w-full h-auto object-contain">
            </div>
            <div class="lg:col-span-5 space-y-3">
              <h5 class="text-xs font-bold text-slate-300 uppercase tracking-wider">Underwriting Approval Summary</h5>
              <div class="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2 text-xs">
                <div class="text-emerald-400 font-semibold">Immediate Execution Authorized:</div>
                <p class="text-slate-300">
                  Ultra-short 4-month tenure and strong employment history heavily mitigated risk, pulling log-odds down by -1.82. Auto-approved with standard interest rate.
                </p>
              </div>
            </div>
          </div>
        </div>

        <div id="case-study-uncertain" class="hidden glass-card rounded-xl p-6 space-y-4 border border-amber-500/30">
          <div class="flex flex-col md:flex-row md:items-center justify-between gap-2 border-b border-slate-800 pb-3">
            <div>
              <span class="text-xs text-amber-400 font-bold uppercase tracking-wider">Tier 2: Senior Underwriter Triage</span>
              <h4 class="text-lg font-bold text-white">Applicant #3 &bull; Calibrated Default Risk = 46.6%</h4>
            </div>
            <div class="text-xs text-slate-400">
              Profile: 18 months, 3,104 DM, Ambiguous Credit History, Moderate Savings
            </div>
          </div>
          <div class="grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">
            <div class="lg:col-span-7 rounded-lg overflow-hidden border border-slate-800 bg-slate-950">
              <img src="__IMG_UNCERTAIN__" alt="SHAP Waterfall Uncertain" class="w-full h-auto object-contain">
            </div>
            <div class="lg:col-span-5 space-y-3">
              <h5 class="text-xs font-bold text-slate-300 uppercase tracking-wider">Underwriter Queue Action</h5>
              <div class="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2 text-xs">
                <div class="text-amber-400 font-semibold">High Model Uncertainty:</div>
                <p class="text-slate-300">
                  Because default risk sits at 46.6% (between 30% and 65%), automated rules step aside. Routed to Senior Underwriter queue for manual verification of tax returns and paystubs.
                </p>
              </div>
            </div>
          </div>
        </div>

      </div>

      <div class="glass-card rounded-xl p-5 space-y-3">
        <div class="flex items-center justify-between">
          <h3 class="text-sm font-bold text-white">SHAP Stability Across 5 Cross-Validation Folds</h3>
          <span class="text-[10px] text-indigo-400 bg-indigo-500/10 border border-indigo-500/20 px-2 py-0.5 rounded">Validation Metric</span>
        </div>
        <p class="text-xs text-slate-400">
          A crucial teacher requirement: Do SHAP rankings change randomly across different training splits? The heatmaps and Spearman correlation (~0.94) prove that feature importance is rock-solid and stable.
        </p>
        <div class="rounded-lg overflow-hidden border border-slate-800 bg-slate-950 max-h-96 flex items-center justify-center">
          <img src="__IMG_STABILITY__" alt="SHAP Stability" class="w-auto h-80 object-contain">
        </div>
      </div>

    </div>

    <!-- TAB 6: CODE ARCHITECTURE WALKTHROUGH -->
    <div id="tab-content-code" class="hidden space-y-8">
      
      <div class="glass-card rounded-2xl p-6 border border-blue-500/20 bg-gradient-to-r from-blue-950/20 via-slate-900 to-slate-950 space-y-3">
        <span class="inline-flex items-center gap-1.5 px-3 py-0.5 rounded-full text-xs font-semibold bg-blue-500/20 text-blue-300 border border-blue-500/30">
          <span>💻</span> Production Python Architecture
        </span>
        <h2 class="text-2xl font-bold text-white">How the Source Code is Structured & How It Runs</h2>
        <p class="text-xs text-slate-400 max-w-3xl leading-relaxed">
          The project is engineered into modular, clean Python files inside <code class="text-indigo-300">src/</code> and accompanied by the full interactive lab notebook <code class="text-indigo-300">notebooks/Credit_Default_Risk_Prediction_SHAP.ipynb</code>.
        </p>
      </div>

      <div class="flex items-center gap-2 overflow-x-auto pb-2 border-b border-slate-800 text-xs">
        <button onclick="switchCodeTab('preproc')" id="btn-code-preproc" class="px-4 py-2 rounded-lg bg-indigo-600 text-white font-medium">1. preprocessing.py</button>
        <button onclick="switchCodeTab('models')" id="btn-code-models" class="px-4 py-2 rounded-lg bg-slate-800 text-slate-400 hover:text-white font-medium">2. models.py</button>
        <button onclick="switchCodeTab('calib')" id="btn-code-calib" class="px-4 py-2 rounded-lg bg-slate-800 text-slate-400 hover:text-white font-medium">3. calibration.py</button>
        <button onclick="switchCodeTab('xai')" id="btn-code-xai" class="px-4 py-2 rounded-lg bg-slate-800 text-slate-400 hover:text-white font-medium">4. explainability.py</button>
        <button onclick="switchCodeTab('pipeline')" id="btn-code-pipeline" class="px-4 py-2 rounded-lg bg-slate-800 text-slate-400 hover:text-white font-medium">5. pipeline.py</button>
      </div>

      <div id="code-snippet-preproc" class="space-y-4">
        <div class="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-2">
          <h4 class="text-sm font-bold text-white">Data Leakage Prevention & ColumnTransformer</h4>
          <p class="text-xs text-slate-400">
            Notice how <code class="text-indigo-300">fit_transform</code> is only called on the training data. The test data is only transformed using the learned parameters.
          </p>
        </div>
        <pre class="glass-card rounded-xl p-5 overflow-x-auto text-xs text-slate-300 border border-slate-800"><code class="language-python"># src/preprocessing.py
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

class CreditDataPreprocessor:
    def __init__(self):
        # Numerical Pipeline: Median imputation + Z-Score standardization
        self.num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical Pipeline: Most-frequent imputation + One-Hot Encoding
        self.cat_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        self.column_transformer = ColumnTransformer(
            transformers=[
                ('num', self.num_pipeline, NUMERICAL_FEATURES),
                ('cat', self.cat_pipeline, CATEGORICAL_FEATURES)
            ]
        )

    def fit_transform(self, X_train):
        # Strictly fit statistics only on the training set!
        return self.column_transformer.fit_transform(X_train)

    def transform(self, X_test):
        # Apply fitted statistics to test set without peeking
        return self.column_transformer.transform(X_test)</code></pre>
      </div>

      <div id="code-snippet-models" class="hidden space-y-4">
        <div class="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-2">
          <h4 class="text-sm font-bold text-white">Model Tuning with Stratified 5-Fold Cross-Validation</h4>
          <p class="text-xs text-slate-400">
            We use <code class="text-indigo-300">scale_pos_weight=2.33</code> for XGBoost and <code class="text-indigo-300">class_weight='balanced'</code> for Random Forest to tackle class imbalance.
          </p>
        </div>
        <pre class="glass-card rounded-xl p-5 overflow-x-auto text-xs text-slate-300 border border-slate-800"><code class="language-python"># src/models.py
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

def get_xgb_pipeline():
    # Scale pos weight = 700 / 300 = 2.333 (penalty for missing defaults)
    xgb = XGBClassifier(
        scale_pos_weight=2.333,
        eval_metric='logloss',
        random_state=42
    )
    param_grid = {
        'max_depth': [3, 4, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200],
        'subsample': [0.8, 1.0]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return GridSearchCV(xgb, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)</code></pre>
      </div>

      <div id="code-snippet-calib" class="hidden space-y-4">
        <div class="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-2">
          <h4 class="text-sm font-bold text-white">Probability Calibration via Platt Scaling</h4>
          <p class="text-xs text-slate-400">
            Using Scikit-Learn's <code class="text-indigo-300">CalibratedClassifierCV(method='sigmoid', cv=5)</code> to map uncalibrated tree margins to true probabilities.
          </p>
        </div>
        <pre class="glass-card rounded-xl p-5 overflow-x-auto text-xs text-slate-300 border border-slate-800"><code class="language-python"># src/calibration.py
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

def calibrate_model(base_estimator, X_train, y_train, method='sigmoid', cv=5):
    # Fits Platt sigmoid calibration curves across internal CV folds
    calibrated = CalibratedClassifierCV(
        estimator=base_estimator,
        method=method,
        cv=cv
    )
    calibrated.fit(X_train, y_train)
    return calibrated

def compute_calibration_metrics(y_true, p_uncal, p_cal):
    brier_uncal = brier_score_loss(y_true, p_uncal)
    brier_cal = brier_score_loss(y_true, p_cal)
    improvement = ((brier_uncal - brier_cal) / brier_uncal) * 100
    return {
        'brier_uncal': brier_uncal,
        'brier_cal': brier_cal,
        'improvement_pct': improvement
    }</code></pre>
      </div>

      <div id="code-snippet-xai" class="hidden space-y-4">
        <div class="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-2">
          <h4 class="text-sm font-bold text-white">TreeExplainer & Regulatory Adverse Action Generation</h4>
          <p class="text-xs text-slate-400">
            Extracts top SHAP factors and maps encoded feature names to official human-readable disclosure statements.
          </p>
        </div>
        <pre class="glass-card rounded-xl p-5 overflow-x-auto text-xs text-slate-300 border border-slate-800"><code class="language-python"># src/explainability.py
import shap

class CreditRiskExplainer:
    def __init__(self, model, feature_names):
        self.explainer = shap.TreeExplainer(model)
        self.feature_names = feature_names

    def explain_instance(self, sample_row):
        # Calculate exact Shapley values for this specific applicant
        shap_values = self.explainer(sample_row)
        
        # Sort factors pushing risk upwards (+ positive SHAP)
        risk_drivers = []
        for feat_name, val, impact in zip(self.feature_names, sample_row, shap_values.values[0]):
            if impact > 0:
                risk_drivers.append((feat_name, impact, val))
        
        risk_drivers.sort(key=lambda x: x[1], reverse=True)
        top_reasons = risk_drivers[:4] # Legally required Top 4 Adverse Reasons
        return top_reasons</code></pre>
      </div>

      <div id="code-snippet-pipeline" class="hidden space-y-4">
        <div class="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-2">
          <h4 class="text-sm font-bold text-white">End-to-End Orchestrator (run_pipeline)</h4>
          <p class="text-xs text-slate-400">
            Runs all 7 stages seamlessly: ingestion &rarr; preprocessing &rarr; tuning &rarr; calibration &rarr; evaluation &rarr; SHAP &rarr; figure output.
          </p>
        </div>
        <pre class="glass-card rounded-xl p-5 overflow-x-auto text-xs text-slate-300 border border-slate-800"><code class="language-python"># src/pipeline.py
def run_pipeline():
    # 1. Load Data
    df_raw = load_raw_data(RAW_DATA_PATH)
    X_train_raw, X_test_raw, y_train, y_test, metadata = prepare_data(df_raw)
    
    # 2. Preprocess
    preprocessor = CreditDataPreprocessor()
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    
    # 3. Model Training & Cross-Validation
    models_dict = tune_and_train_models(X_train, y_train)
    
    # 4. Platt Calibration
    calibrated_models = {name: calibrate_model(m, X_train, y_train) for name, m in models_dict.items()}
    
    # 5. Evaluate & Generate Figures
    eval_df = evaluate_models(calibrated_models, X_test, y_test)
    
    # 6. SHAP Analysis
    explainer = CreditRiskExplainer(calibrated_models['XGBoost'], preprocessor.feature_names_)
    explainer.generate_all_reports()
    print("Pipeline Complete! All figures generated in reports/figures/")</code></pre>
      </div>

    </div>

    <!-- TAB 7: VIVA & EXAM DEFENSE PREP -->
    <div id="tab-content-viva" class="hidden space-y-8">
      
      <div class="glass-card rounded-2xl p-6 border border-emerald-500/20 bg-gradient-to-r from-emerald-950/20 via-slate-900 to-slate-950 space-y-3">
        <span class="inline-flex items-center gap-1.5 px-3 py-0.5 rounded-full text-xs font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
          <span>🎯</span> University Exam & Viva Defense
        </span>
        <h2 class="text-2xl font-bold text-white">The Top 10 Questions Your Teacher Will Ask (With Answers)</h2>
        <p class="text-xs text-slate-400 max-w-3xl leading-relaxed">
          Click any card to reveal the exact, polished technical response you should give to your examiner.
        </p>
      </div>

      <div class="space-y-4">
        
        <div class="glass-card rounded-xl p-5 border border-slate-800 cursor-pointer hover:border-indigo-500/50 transition-all" onclick="toggleViva(1)">
          <div class="flex items-center justify-between">
            <h3 class="text-sm font-bold text-white flex items-center gap-2">
              <span class="text-indigo-400">Q1.</span> Why did you use ROC-AUC and PR-AUC instead of simple Accuracy?
            </h3>
            <span id="viva-icon-1" class="text-slate-400 font-bold">&plus;</span>
          </div>
          <div id="viva-ans-1" class="hidden mt-4 pt-3 border-t border-slate-800 text-xs text-slate-300 space-y-2 leading-relaxed">
            <p><strong>Perfect Answer</strong>: <em>"In credit risk, classes are heavily imbalanced (70% good, 30% bad). If a model simply predicts 'Good' for every applicant, it gets 70% accuracy while failing to detect 100% of defaults! ROC-AUC measures discriminatory ability across all thresholds, and PR-AUC explicitly measures precision and recall on the rare default class. Our champion XGBoost achieved the highest PR-AUC of 0.665."</em></p>
          </div>
        </div>

        <div class="glass-card rounded-xl p-5 border border-slate-800 cursor-pointer hover:border-indigo-500/50 transition-all" onclick="toggleViva(2)">
          <div class="flex items-center justify-between">
            <h3 class="text-sm font-bold text-white flex items-center gap-2">
              <span class="text-indigo-400">Q2.</span> What is Data Leakage and how did you prevent it in your code?
            </h3>
            <span id="viva-icon-2" class="text-slate-400 font-bold">&plus;</span>
          </div>
          <div id="viva-ans-2" class="hidden mt-4 pt-3 border-t border-slate-800 text-xs text-slate-300 space-y-2 leading-relaxed">
            <p><strong>Perfect Answer</strong>: <em>"Data leakage happens when information from outside the training dataset is inadvertently used to create the model—for example, calculating the mean, median, or scaling statistics across all 1,000 applicants before splitting. We strictly performed a Stratified 80/20 train/test split first, and fitted our imputers, scalers, and encoders exclusively on X_train. The test set was transformed using only training statistics."</em></p>
          </div>
        </div>

        <div class="glass-card rounded-xl p-5 border border-slate-800 cursor-pointer hover:border-indigo-500/50 transition-all" onclick="toggleViva(3)">
          <div class="flex items-center justify-between">
            <h3 class="text-sm font-bold text-white flex items-center gap-2">
              <span class="text-indigo-400">Q3.</span> Why do tree models need Platt Scaling? Don't they already output probabilities?
            </h3>
            <span id="viva-icon-3" class="text-slate-400 font-bold">&plus;</span>
          </div>
          <div id="viva-ans-3" class="hidden mt-4 pt-3 border-t border-slate-800 text-xs text-slate-300 space-y-2 leading-relaxed">
            <p><strong>Perfect Answer</strong>: <em>"Gradient boosted trees output leaf scores that are excellent for ranking relative risk, but uncalibrated in a statistical sense; they tend to push probabilities toward extreme values or cluster in narrow ranges. Platt scaling fits a cross-validated logistic sigmoid mapping raw margins to empirical default rates. This reduced our Brier score from 0.1740 to 0.1570 (a 9.7% improvement), ensuring that a 30% score actually means 3 out of 10 people default."</em></p>
          </div>
        </div>

        <div class="glass-card rounded-xl p-5 border border-slate-800 cursor-pointer hover:border-indigo-500/50 transition-all" onclick="toggleViva(4)">
          <div class="flex items-center justify-between">
            <h3 class="text-sm font-bold text-white flex items-center gap-2">
              <span class="text-indigo-400">Q4.</span> What is the difference between SMOTE and Class-Weighted Loss? Which was better?
            </h3>
            <span id="viva-icon-4" class="text-slate-400 font-bold">&plus;</span>
          </div>
          <div id="viva-ans-4" class="hidden mt-4 pt-3 border-t border-slate-800 text-xs text-slate-300 space-y-2 leading-relaxed">
            <p><strong>Perfect Answer</strong>: <em>"SMOTE synthesizes new artificial minority points using k-nearest neighbors in feature space before training. Class-weighted loss modifies the algorithm's loss function directly by penalizing minority misclassifications with a higher multiplier (w1 = 2.33). In our experiments, Class-Weighted XGBoost achieved a higher PR-AUC (0.665 vs 0.651) with less distortion of feature distributions than SMOTE."</em></p>
          </div>
        </div>

        <div class="glass-card rounded-xl p-5 border border-slate-800 cursor-pointer hover:border-indigo-500/50 transition-all" onclick="toggleViva(5)">
          <div class="flex items-center justify-between">
            <h3 class="text-sm font-bold text-white flex items-center gap-2">
              <span class="text-indigo-400">Q5.</span> Why is SHAP superior to standard Gini feature importance?
            </h3>
            <span id="viva-icon-5" class="text-slate-400 font-bold">&plus;</span>
          </div>
          <div id="viva-ans-5" class="hidden mt-4 pt-3 border-t border-slate-800 text-xs text-slate-300 space-y-2 leading-relaxed">
            <p><strong>Perfect Answer</strong>: <em>"Standard Gini feature importance only provides a macro-level metric of how frequently a feature was split on, and it is heavily biased toward continuous variables with high cardinality. Most importantly, Gini cannot explain individual predictions. SHAP is grounded in cooperative game theory (satisfying Efficiency, Symmetry, and Additivity axioms) and provides local waterfall attributions for specific applicants, enabling automated legal Adverse Action disclosures."</em></p>
          </div>
        </div>

        <div class="glass-card rounded-xl p-5 border border-slate-800 cursor-pointer hover:border-indigo-500/50 transition-all" onclick="toggleViva(6)">
          <div class="flex items-center justify-between">
            <h3 class="text-sm font-bold text-white flex items-center gap-2">
              <span class="text-indigo-400">Q6.</span> Explain the 3-Tier Underwriting Decision Policy.
            </h3>
            <span id="viva-icon-6" class="text-slate-400 font-bold">&plus;</span>
          </div>
          <div id="viva-ans-6" class="hidden mt-4 pt-3 border-t border-slate-800 text-xs text-slate-300 space-y-2 leading-relaxed">
            <p><strong>Perfect Answer</strong>: <em>"Rather than an arbitrary 0.50 binary threshold, we establish three operational tiers:
            1. P &lt; 0.30: Tier 1 Auto-Approve (Low risk, instant fulfillment).
            2. 0.30 &le; P &le; 0.65: Tier 2 Human Review (High model uncertainty zone, routed to a human underwriter).
            3. P &gt; 0.65: Tier 3 Auto-Decline (High risk, automated FCRA Adverse Action Statement generated with top 4 SHAP drivers)."</em></p>
          </div>
        </div>

      </div>

      <div class="glass-card rounded-2xl p-6 border border-indigo-500/30 space-y-4">
        <h3 class="text-base font-bold text-white flex items-center gap-2">
          <span>📝</span> 3-Minute Knowledge Check
        </h3>
        <p class="text-xs text-slate-400">Test if you're ready for your lab evaluation:</p>

        <div class="space-y-4 text-xs">
          <div class="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-2">
            <p class="font-semibold text-slate-200">1. If you calculate the imputer median on the whole dataset before train_test_split, what is this error called?</p>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-2">
              <button onclick="checkQuiz(this, false)" class="p-2.5 rounded-lg bg-slate-800 text-left hover:bg-slate-700 text-slate-300">A) Overfitting</button>
              <button onclick="checkQuiz(this, true)" class="p-2.5 rounded-lg bg-slate-800 text-left hover:bg-slate-700 text-slate-300">B) Data Leakage (Correct!)</button>
              <button onclick="checkQuiz(this, false)" class="p-2.5 rounded-lg bg-slate-800 text-left hover:bg-slate-700 text-slate-300">C) Class Imbalance</button>
              <button onclick="checkQuiz(this, false)" class="p-2.5 rounded-lg bg-slate-800 text-left hover:bg-slate-700 text-slate-300">D) Model Drift</button>
            </div>
          </div>

          <div class="p-4 rounded-xl bg-slate-900 border border-slate-800 space-y-2">
            <p class="font-semibold text-slate-200">2. What metric assesses whether predicted probabilities match true default rates?</p>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-2">
              <button onclick="checkQuiz(this, true)" class="p-2.5 rounded-lg bg-slate-800 text-left hover:bg-slate-700 text-slate-300">A) Brier Score (Correct!)</button>
              <button onclick="checkQuiz(this, false)" class="p-2.5 rounded-lg bg-slate-800 text-left hover:bg-slate-700 text-slate-300">B) ROC-AUC</button>
              <button onclick="checkQuiz(this, false)" class="p-2.5 rounded-lg bg-slate-800 text-left hover:bg-slate-700 text-slate-300">C) Accuracy</button>
              <button onclick="checkQuiz(this, false)" class="p-2.5 rounded-lg bg-slate-800 text-left hover:bg-slate-700 text-slate-300">D) Precision</button>
            </div>
          </div>
        </div>
      </div>

    </div>

  </main>

  <footer class="glass-card border-t border-slate-800 px-6 py-4 mt-12 text-xs text-slate-500">
    <div class="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-2">
      <div>University ML Lab Project &bull; Credit Default Risk Prediction using SHAP & Tree Ensembles</div>
      <div class="flex items-center gap-4">
        <span>Files: <code class="text-indigo-400">notebooks/</code> &bull; <code class="text-indigo-400">src/</code> &bull; <code class="text-indigo-400">reports/</code></span>
      </div>
    </div>
  </footer>

  <script>
    function switchTab(tabId) {
      const tabs = ['overview', 'simulator', 'theory', 'benchmark', 'shap', 'code', 'viva'];
      tabs.forEach(t => {
        const btn = document.getElementById('tab-btn-' + t);
        const content = document.getElementById('tab-content-' + t);
        if (t === tabId) {
          btn.classList.add('active');
          btn.classList.remove('text-slate-400');
          btn.classList.add('text-white');
          content.classList.remove('hidden');
        } else {
          btn.classList.remove('active');
          btn.classList.add('text-slate-400');
          btn.classList.remove('text-white');
          content.classList.add('hidden');
        }
      });
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }

    function switchCaseStudy(caseId) {
      const cases = ['declined', 'approved', 'uncertain'];
      cases.forEach(c => {
        const btn = document.getElementById('btn-case-' + c);
        const content = document.getElementById('case-study-' + c);
        if (c === caseId) {
          btn.className = 'px-3 py-1.5 rounded-lg text-xs font-semibold bg-indigo-600 text-white shadow-lg';
          content.classList.remove('hidden');
        } else {
          btn.className = 'px-3 py-1.5 rounded-lg text-xs font-semibold bg-slate-800 text-slate-300 hover:text-white';
          content.classList.add('hidden');
        }
      });
    }

    function switchCodeTab(codeId) {
      const codes = ['preproc', 'models', 'calib', 'xai', 'pipeline'];
      codes.forEach(c => {
        const btn = document.getElementById('btn-code-' + c);
        const snippet = document.getElementById('code-snippet-' + c);
        if (c === codeId) {
          btn.className = 'px-4 py-2 rounded-lg bg-indigo-600 text-white font-medium';
          snippet.classList.remove('hidden');
        } else {
          btn.className = 'px-4 py-2 rounded-lg bg-slate-800 text-slate-400 hover:text-white font-medium';
          snippet.classList.add('hidden');
        }
      });
    }

    function toggleViva(id) {
      const ans = document.getElementById('viva-ans-' + id);
      const icon = document.getElementById('viva-icon-' + id);
      if (ans.classList.contains('hidden')) {
        ans.classList.remove('hidden');
        icon.innerHTML = '&minus;';
      } else {
        ans.classList.add('hidden');
        icon.innerHTML = '&plus;';
      }
    }

    function checkQuiz(btn, isCorrect) {
      if (isCorrect) {
        btn.classList.remove('bg-slate-800', 'text-slate-300');
        btn.classList.add('bg-emerald-600', 'text-white', 'font-bold');
      } else {
        btn.classList.remove('bg-slate-800', 'text-slate-300');
        btn.classList.add('bg-rose-600', 'text-white', 'opacity-60');
      }
    }

    function updateSimulator() {
      const duration = parseInt(document.getElementById('input-duration').value);
      const amount = parseInt(document.getElementById('input-amount').value);
      const checking = document.getElementById('input-checking').value;
      const savings = document.getElementById('input-savings').value;
      const age = parseInt(document.getElementById('input-age').value);
      const housing = document.getElementById('input-housing').value;
      const employment = document.getElementById('input-employment').value;

      document.getElementById('val-duration').innerText = duration + ' months';
      document.getElementById('val-amount').innerText = amount.toLocaleString() + ' DM';
      document.getElementById('val-age').innerText = age + ' yrs';

      let baseLogOdds = -0.85;
      let shapDrivers = [];

      let shapDuration = ((duration - 20) / 10) * 0.28;
      shapDrivers.push({
        code: 'duration',
        name: 'Loan Duration (' + duration + ' mos)',
        impact: shapDuration,
        reason: 'Duration of requested loan term is excessively long relative to risk profile.'
      });

      let shapAmount = ((amount - 3000) / 3000) * 0.18;
      shapDrivers.push({
        code: 'credit_amount',
        name: 'Credit Amount (' + amount.toLocaleString() + ' DM)',
        impact: shapAmount,
        reason: 'Requested credit amount is high relative to borrower repayment capacity.'
      });

      let shapChecking = 0;
      let checkingReason = 'Status of checking account is low, overdrawn, or absent.';
      if (checking === 'lt_0') shapChecking = 0.58;
      else if (checking === '0_to_200') shapChecking = 0.22;
      else if (checking === 'ge_200') shapChecking = -0.25;
      else if (checking === 'no_checking') shapChecking = -0.42;
      shapDrivers.push({
        code: 'checking_status',
        name: 'Checking Account Balance',
        impact: shapChecking,
        reason: checkingReason
      });

      let shapSavings = 0;
      let savingsReason = 'Insufficient liquid reserves or low balance in savings/investment accounts.';
      if (savings === 'lt_100') shapSavings = 0.32;
      else if (savings === '100_to_500') shapSavings = 0.08;
      else if (savings === '500_to_1000') shapSavings = -0.15;
      else if (savings === 'ge_1000') shapSavings = -0.38;
      else if (savings === 'unknown') shapSavings = -0.05;
      shapDrivers.push({
        code: 'savings_status',
        name: 'Savings Reserves',
        impact: shapSavings,
        reason: savingsReason
      });

      let shapAge = ((35 - age) / 15) * 0.16;
      shapDrivers.push({
        code: 'age',
        name: 'Borrower Age (' + age + ' yrs)',
        impact: shapAge,
        reason: 'Age and short financial history relative to requested obligation.'
      });

      let shapHousing = 0;
      if (housing === 'rent') shapHousing = 0.14;
      else if (housing === 'own') shapHousing = -0.20;
      else if (housing === 'for_free') shapHousing = 0.10;
      shapDrivers.push({
        code: 'housing',
        name: 'Housing (' + housing + ')',
        impact: shapHousing,
        reason: 'Residential tenancy status increases exposure to financial shocks.'
      });

      let shapEmp = 0;
      if (employment === 'unemployed') shapEmp = 0.40;
      else if (employment === 'lt_1') shapEmp = 0.18;
      else if (employment === '1_to_4') shapEmp = 0.02;
      else if (employment === '4_to_7') shapEmp = -0.14;
      else if (employment === 'ge_7') shapEmp = -0.30;
      shapDrivers.push({
        code: 'employment',
        name: 'Employment Stability',
        impact: shapEmp,
        reason: 'Short employment tenure or current employment disruption.'
      });

      let totalLogOdds = baseLogOdds;
      shapDrivers.forEach(d => { totalLogOdds += d.impact; });

      let calibratedProb = 1 / (1 + Math.exp(- (1.12 * totalLogOdds - 0.08)));
      if (calibratedProb < 0.03) calibratedProb = 0.03;
      if (calibratedProb > 0.96) calibratedProb = 0.96;

      const pct = (calibratedProb * 100).toFixed(1);
      document.getElementById('sim-prob-text').innerText = pct + '%';
      document.getElementById('sim-prob-bar').style.width = pct + '%';

      const card = document.getElementById('sim-decision-card');
      const badge = document.getElementById('sim-tier-badge');
      const actionText = document.getElementById('sim-action-text');
      const probText = document.getElementById('sim-prob-text');
      const adverseBox = document.getElementById('sim-adverse-box');

      if (calibratedProb < 0.30) {
        badge.className = 'px-3 py-1 rounded-full text-xs font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/40';
        badge.innerText = 'TIER 1: AUTO-APPROVE';
        probText.className = 'text-3xl font-extrabold text-emerald-400 font-mono';
        card.className = 'glass-card rounded-2xl p-6 border border-emerald-500/40 bg-gradient-to-br from-emerald-950/20 to-slate-900 space-y-4';
        actionText.innerText = 'Default risk is strictly under 30%. Application is approved immediately with competitive interest rate.';
        adverseBox.classList.add('hidden');
      } else if (calibratedProb <= 0.65) {
        badge.className = 'px-3 py-1 rounded-full text-xs font-bold bg-amber-500/20 text-amber-300 border border-amber-500/40';
        badge.innerText = 'TIER 2: HUMAN REVIEW';
        probText.className = 'text-3xl font-extrabold text-amber-400 font-mono';
        card.className = 'glass-card rounded-2xl p-6 border border-amber-500/40 bg-gradient-to-br from-amber-950/20 to-slate-900 space-y-4';
        actionText.innerText = 'Default risk is in the 30% - 65% high-uncertainty zone. Automated decision is suspended; application is routed to Senior Underwriter for collateral evaluation.';
        adverseBox.classList.remove('hidden');
      } else {
        badge.className = 'px-3 py-1 rounded-full text-xs font-bold bg-rose-500/20 text-rose-300 border border-rose-500/40';
        badge.innerText = 'TIER 3: AUTO-DECLINE';
        probText.className = 'text-3xl font-extrabold text-rose-400 font-mono';
        card.className = 'glass-card rounded-2xl p-6 border border-rose-500/40 bg-gradient-to-br from-rose-950/20 to-slate-900 space-y-4';
        actionText.innerText = 'Default risk exceeds the 65% threshold. Application is declined; federal regulations require issuing an Adverse Action notice detailing the top reasons.';
        adverseBox.classList.remove('hidden');
      }

      shapDrivers.sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact));
      const container = document.getElementById('sim-shap-bars');
      container.innerHTML = '';

      shapDrivers.forEach(d => {
        const isPos = d.impact >= 0;
        const widthPct = Math.min(Math.abs(d.impact) * 85, 100);
        const color = isPos ? 'bg-rose-500' : 'bg-emerald-500';
        const sign = isPos ? '+' : '';
        const textColor = isPos ? 'text-rose-400' : 'text-emerald-400';

        const row = document.createElement('div');
        row.className = 'space-y-1';
        row.innerHTML = `
          <div class="flex justify-between text-[11px]">
            <span class="text-slate-300 font-medium">${d.name}</span>
            <span class="font-mono font-bold ${textColor}">${sign}${d.impact.toFixed(2)} log-odds</span>
          </div>
          <div class="w-full bg-slate-800 h-2 rounded-full overflow-hidden flex">
            <div class="${color} h-full transition-all duration-300 rounded-full" style="width: ${widthPct}%;"></div>
          </div>
        `;
        container.appendChild(row);
      });

      const adverseList = document.getElementById('sim-adverse-list');
      adverseList.innerHTML = '';
      const adverseFactors = shapDrivers.filter(d => d.impact > 0).slice(0, 4);

      if (adverseFactors.length === 0) {
        adverseList.innerHTML = '<li class="text-slate-400">No adverse risk drivers detected.</li>';
      } else {
        adverseFactors.forEach(f => {
          const li = document.createElement('li');
          li.innerHTML = `<strong>${f.name}</strong>: ${f.reason}`;
          adverseList.appendChild(li);
        });
      }
    }

    function resetSimulator() {
      document.getElementById('input-duration').value = 48;
      document.getElementById('input-amount').value = 3915;
      document.getElementById('input-checking').value = 'lt_0';
      document.getElementById('input-savings').value = 'lt_100';
      document.getElementById('input-age').value = 25;
      document.getElementById('input-housing').value = 'rent';
      document.getElementById('input-employment').value = 'lt_1';
      updateSimulator();
    }

    window.addEventListener('DOMContentLoaded', () => {
      updateSimulator();
    });
  </script>
</body>
</html>
"""

    # Replace placeholders with base64 images
    html_content = template.replace("__IMG_ROC_PR__", img_roc_pr)
    html_content = html_content.replace("__IMG_CAL__", img_cal)
    html_content = html_content.replace("__IMG_MODEL_COMP__", img_model_comp)
    html_content = html_content.replace("__IMG_BEESWARM__", img_shap_beeswarm)
    html_content = html_content.replace("__IMG_SUMMARY__", img_shap_summary)
    html_content = html_content.replace("__IMG_DECLINED__", img_shap_declined)
    html_content = html_content.replace("__IMG_APPROVED__", img_shap_approved)
    html_content = html_content.replace("__IMG_UNCERTAIN__", img_shap_uncertain)
    html_content = html_content.replace("__IMG_STABILITY__", img_shap_stability)
    html_content = html_content.replace("__IMG_UNCERTAINTY_DIST__", img_uncertainty)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Mastery Dashboard generated successfully at: {OUTPUT_PATH}")

    with open(REPORT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Report copy generated successfully at: {REPORT_OUTPUT_PATH}")

if __name__ == "__main__":
    build_dashboard()
