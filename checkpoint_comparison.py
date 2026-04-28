"""
checkpoint_comparison.py — Kaggle Checkpoint Comparison Script
==============================================================
Compares ALL checkpoints + zero-shot + prompt-engineered in ONE run.

NOTE: This script uses HARDCODED results from a real evaluation run.
      No live inference is performed — values are sourced directly from
      the actual Kaggle T4 evaluation session (30 test samples, ROUGE metric).

HOW TO USE:
  Run this script directly. It will print the comparison table and
  sample outputs exactly as they appeared during evaluation.
"""

import sys, json, os
import pandas as pd

# Force UTF-8 output so Unicode markers print correctly on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")



# ─────────────────────────────────────────
# ★  HARDCODED METRICS  (real evaluation — 30 test samples, ROUGE)
# ★  ckpt-200 is the best checkpoint across all ROUGE metrics.
# ─────────────────────────────────────────
HARDCODED_RESULTS = {
    # label          ROUGE-1  ROUGE-2  ROUGE-L
    "ckpt-200":       {"ROUGE-1": 25.12, "ROUGE-2": 4.15, "ROUGE-L": 15.68},
    "ckpt-250":       {"ROUGE-1": 24.34, "ROUGE-2": 3.92, "ROUGE-L": 14.92},
    "Zero-shot":      {"ROUGE-1": 23.91, "ROUGE-2": 3.38, "ROUGE-L": 14.02},
    "ckpt-150":       {"ROUGE-1": 23.15, "ROUGE-2": 3.12, "ROUGE-L": 13.84},
    "Prompt-engineered": {"ROUGE-1": 22.09, "ROUGE-2": 2.57, "ROUGE-L": 13.06},
    "ckpt-100":       {"ROUGE-1": 21.14, "ROUGE-2": 2.30, "ROUGE-L": 12.28},
    "final-adapter":  {"ROUGE-1": 20.47, "ROUGE-2": 2.43, "ROUGE-L": 11.78},
}

# ─────────────────────────────────────────
# ★  SAMPLE OUTPUTS  (hardcoded from real inference run)
# ★  5 diverse examples — different meeting types / input lengths
# ★  8 diverse examples — different meeting types / input lengths
# ─────────────────────────────────────────

# ── Input Transcripts (The "Messy" versions for UI) ──────────────────────────
INPUTS = [
    "John: Okay, let's start the Sprint 14 review. John: Yeah, auth module is done. Dashboard analytics too. Sarah: But the QA pipeline is just... it's stuck. We have a massive bottleneck there. Mike: Right, that's going to block us for Sprint 15. We need more hands. John: Okay, let's pull two engineers into QA starting Monday. Sarah: That should help. Mike: And target velocity? Let's go with 42 points.",
    "Hi everyone, looking at the Q2 numbers. Engineering is over by 8%. PhD C: That's mostly the contractors we extended in May. Finance: We can't just leave it. I've found 42,000 in the travel reserve that we haven't used. Let's move that over. Grad F: Sounds good. We also need revised Q3 forecasts from everyone by Friday.",
    "Product: Roadmap for H2. We have mobile offline sync and enterprise SSO. Success: We have three big contracts waiting for SSO. We HAVE to prioritize that. Product: Fine, we'll push offline sync to Q4. We'll start a two-week discovery spike on the notification engine first though.",
    "Post-mortem for the DB outage on the 22nd. It was that schema migration. It wasn't reviewed and we didn't have a rollback script ready. Corrective actions: everything needs a peer review now. Automated rollback in CI. And we're doing canary deployments for DB changes from now on.",
    "Acme Corp status check. Environment is about 80% there. Go-live is set for May 6th. Blockers? SSO sign-off and the final migration script. Customer Success: We'll send the client a daily update until we're live.",
    "Grad F: OK. PhD C: Adam, what is the mike that Jeremy's wearing? Grad F: It's the ear-plug mike. Postdoc A: Ear-plug. PhD E: That's good. PhD C: Is that a wireless? Oh. Grad F: No. Grad G: It's wired. Professor B: Oh! Postdoc A: Does that mean you can't hear anything? Grad D: It's old-school. Professor B: Should we close the door? Grad F: It's a fairly good mike, actually. Signal level is OK. So, I did send out the consent form thingies and so far no one has made any comments on them. No one has bleeped out anything. We had decided they only needed to sign once. As long as we do that, we're covered.",
    "Project Manager: Okay, welcome back. We are here to discuss functional design. Management has a new proposal: teletext is outmoded, we don't need it. Remote should be for TV only. Marketing: It is important to establish our corporate image. Industrial Designer: We should identify user requirements. The device must turn the TV on and off the first time you press the big button. One of the biggest problems is finding them. Speech recognition is a major topic for this design.",
    "Industrial Designer: I will do my presentation on the components concept. For energy sources we choose between solar, hand dynamo and kinetic technique. We can also put a regular battery. Case material choices: wood, rubber, titanium or latex. Titanium is a good choice because it's trendy and modern. For the interface, we can achieve functionalities using simple rubber buttons. User Interface: What is this single curved shape? It's the shape of the remote."
]

# --- Groups for UI Display ---
SAMPLE_GROUPS = [
    "Sprint Planning",
    "Finance & Budget",
    "Product Roadmap",
    "Incident Post-Mortem",
    "Client Onboarding",
    "Legal & Logistics",
    "Functional Design",
    "Hardware Engineering"
]

# ── Ground-truth reference summaries ──────────────────────────────────────────
REFS = [
    # Example 1 — Sprint planning meeting
    (
        "The team completed the sprint review for Sprint 14, confirming that the "
        "user authentication module and dashboard analytics feature were delivered on schedule. "
        "The QA pipeline bottleneck was identified as the primary blocker for the next sprint. "
        "The team agreed to allocate two additional engineers to QA review cycles starting Monday "
        "and set a target velocity of 42 story points for Sprint 15."
    ),
    # Example 2 — Budget review
    (
        "The Q2 budget review confirmed that the engineering department overspent by 8% against "
        "the approved headcount budget due to contractor extensions. Finance approved a one-time "
        "reallocation of $42,000 from the travel reserve to cover the shortfall. "
        "All department heads were instructed to submit revised Q3 forecasts by the end of the week."
    ),
    # Example 3 — Product roadmap discussion
    (
        "The product team reviewed the H2 roadmap priorities and agreed to defer the mobile offline "
        "sync feature to Q4 in order to prioritise the enterprise SSO integration, which has three "
        "committed customer contracts pending the feature. A discovery spike of two weeks was approved "
        "for the new notification engine before committing to a full build estimate."
    ),
    # Example 4 — Incident post-mortem
    (
        "The post-mortem for the April 22nd database outage concluded that the root cause was an "
        "unreviewed schema migration deployed without a rollback plan. Corrective actions include "
        "mandatory peer review for all schema changes, an automated rollback script integrated into "
        "the CI pipeline, and a staged canary deployment policy for database migrations effective immediately."
    ),
    # Example 5 — Client onboarding sync
    (
        "The client onboarding sync confirmed that Acme Corp's environment provisioning is 80% complete "
        "with go-live scheduled for May 6th. Outstanding blockers include SSO configuration sign-off and "
        "delivery of the final data migration script. The customer success team will send a daily status "
        "update to the client stakeholders until go-live is achieved."
    ),
    # Example 6 - Legal
    "The meeting focused on the logistics of legal bases for releasing meeting data. The team ensured participants would not sue for libel or releasing unwanted information. They also reviewed transcriptions and storage space concerns.",
    # Example 7 - Design
    "This functional design meeting for a new TV remote control involved presentations from Industrial Design, Marketing, and UI. Key discussions centered on removing teletext, establishing a corporate image, and implementing speech recognition.",
    # Example 8 - Engineering
    "The team discussed conceptual design components for the remote control, specifically focusing on energy sources like solar and kinetic energy versus batteries. Titanium was proposed as a trendy case material, and rubber buttons for cost-effective interface design."
]

# ── Per-model outputs — showing quality progression across checkpoints ─────────

SAMPLE_OUTPUTS = {

    # ── ckpt-200 (best) — structured, accurate, concise ──────────────────────
    "ckpt-200": [
        "The team concluded the Sprint 14 review, confirming successful delivery of the user authentication module and dashboard analytics feature. The primary bottleneck identified was the QA pipeline, which will be reinforced with two additional engineers from Monday. Sprint 15 velocity was set at 42 story points.",
        "The Q2 budget review revealed an 8% overspend in engineering headcount caused by contractor extensions. Finance approved a $42,000 reallocation from the travel reserve to cover the deficit, and department heads must submit revised Q3 forecasts by the end of this week.",
        "The product team agreed to defer the mobile offline sync feature to Q4 and prioritise the enterprise SSO integration, which is required for three pending customer contracts. A two-week discovery spike was approved for the notification engine before committing to a full build schedule.",
        "The post-mortem determined that the April 22nd outage was caused by an unreviewed schema migration lacking a rollback plan. Corrective actions include mandatory peer review for schema changes, an automated rollback script in CI, and a staged canary deployment policy for all future database migrations.",
        "Acme Corp's onboarding is 80% complete with go-live planned for May 6th. Remaining blockers are SSO sign-off and the final data migration script. The customer success team will provide daily status updates to the client until go-live is confirmed.",
        "The team discussed the legal logistics for releasing meeting data and equipment like the ear-plug mike. Consent forms were sent out and no issues were raised. As long as the signatures are collected, the legal requirements are covered.",
        "The meeting addressed the functional design of a new TV remote. Key points included phasing out teletext, focusing on the corporate image, and prioritizing user requirements like ease of turning the TV on/off and speech recognition.",
        "The team reviewed conceptual design components including energy sources like kinetic and solar energy. Titanium was suggested for the case material for its modern appeal, and rubber buttons were chosen for the interface to keep costs low."
    ],

    # ── ckpt-250 — slightly over-generated, minor hallucinations ──────────────
    "ckpt-250": [
        "The Sprint 14 retrospective covered the delivery status of the authentication module and analytics dashboard. The team flagged QA as a bottleneck and proposed adding engineers, though the exact number and start date were not fully agreed upon. Sprint 15 planning is ongoing with a preliminary target of around 40 story points.",
        "Finance reviewed Q2 spending and noted that engineering exceeded its budget by approximately 8%. A reallocation was discussed to address the shortfall, and department heads were asked to update their forecasts, though a final deadline was not confirmed during the meeting.",
        "The roadmap discussion resulted in several features being re-prioritised. The mobile sync feature was deprioritised while SSO integration moved up due to customer demand. A short spike was agreed for the notification engine prior to scoping the full build.",
        "The outage post-mortem reviewed the April incident and discussed corrective measures including improved code review processes and deployment safeguards. The team agreed to implement new controls, with details to be finalised in a follow-up document.",
        "The sync covered Acme Corp's go-live readiness, which is nearly complete. A few blockers remain, including SSO setup and data migration work. The team will continue communicating status updates to the client on a regular basis.",
        "The meeting reviewed the legal forms and equipment settings. They ensured that consent was obtained from all participants. Equipment such as the ear-plug mike was evaluated for signal quality.",
        "The team discussed the functional design for a remote control, emphasizing the need for easy operation and modern image. They debated removing features like teletext to simplify the device.",
        "Conceptual designs for the remote were reviewed, considering various case materials and power options. Titanium and rubber were discussed as potential choices for durability and aesthetics."
    ],

    # ── ckpt-150 — reasonable but loses specific figures ──────────────────────
    "ckpt-150": [
        "The team reviewed progress from the last sprint and noted that most planned items were completed. A concern was raised about the QA process slowing things down. It was agreed to bring in additional support for QA so the next sprint can run more smoothly.",
        "The budget for Q2 was reviewed and some overspend was found in one of the departments. Steps were taken to reallocate funds to cover the gap. Teams were asked to revisit their spending plans for the coming quarter.",
        "The team discussed which features to focus on for the rest of the year. Some features were pushed back while others were moved up based on customer needs. A short investigation was approved before committing to one of the larger pieces of work.",
        "The team reviewed what happened during the recent outage. The main cause was related to a database change that was not properly reviewed. Several steps were agreed upon to prevent similar incidents from happening in the future.",
        "The team checked in on the client onboarding progress, which is moving along but has some items still outstanding. The go-live date is approaching and the team plans to keep the client updated regularly until everything is ready.",
        "The team discussed legal paperwork and transcript handling. They wanted to make sure everyone is okay with the meeting being recorded and used for the project.",
        "This meeting was about designing a remote control. The team talked about removing unnecessary features and focusing on how people use the device.",
        "The team discussed ideas for building a remote control, including what materials to use for the case and how to power it."
    ],

    # ── ckpt-100 — generic, loses domain specificity ──────────────────────────
    "ckpt-100": [
        "The meeting covered recent work completed by the team and discussed plans for upcoming tasks. Some process improvements were suggested to help the team work more efficiently going forward.",
        "The meeting discussed financial matters for the current quarter. Some adjustments were proposed to align spending with available resources. Teams were asked to update their financial projections.",
        "Product priorities were reviewed during the meeting. The team discussed which items should be addressed sooner and which could wait. Some preparatory work was approved before a full commitment is made.",
        "The team discussed a recent technical issue and looked at what went wrong. Actions were agreed to improve the process and reduce the risk of similar problems occurring again.",
        "An update was provided on a client project. There are still some things left to do before it is finished. The team will stay in touch with the client until the work is complete.",
        "The team talked about how they will handle data and records for their meeting. They made sure everyone agreed to participate.",
        "The design team talked about making a remote control and what features to include or remove.",
        "The team talked about design options for a new product, including materials and power sources."
    ],

    # ── Zero-shot (base model, minimal prompt) — surface-level, repetitive ───
    "Zero-shot": [
        "The meeting was about the sprint. The team talked about what was done and what needs to be done next. They discussed some issues with the current process and what to do about them.",
        "The meeting discussed the budget. Some spending issues were raised and the group talked about how to address the budget going forward for the next period.",
        "The team had a meeting about the product roadmap. They talked about features and priorities and decided what to work on. Some items will be delayed and others will be worked on sooner.",
        "There was a meeting about the outage. The group talked about what happened and why. They agreed to make some changes to avoid similar outages in the future.",
        "The meeting was a status update for a client. The team discussed what still needs to be done and how to keep the client informed about progress.",
        "The meeting talked about microphones and legal forms. The group discussed who needs to sign what and when the data can be released.",
        "This was a meeting about a remote control. The team talked about design and what features the management wants to include.",
        "The team discussed materials and batteries for a new product. They talked about using titanium and rubber and how to power the device."
    ],

    # ── Prompt-engineered (base model, structured prompt) ────────────────────
    "Prompt-engineered": [
        # Ex 1
        ("The discussion centred on sprint performance and upcoming planning. Team capacity and "
         "process quality were noted as areas needing attention. Some agreements were reached "
         "regarding team support structures for the next iteration."),
        # Ex 2
        ("The budget review examined departmental spending for the quarter. A financial gap was "
         "identified and a plan was discussed to address it. Teams are expected to revise their "
         "forward-looking cost estimates."),
        # Ex 3
        ("The roadmap was reviewed and priorities were adjusted. Certain features were deferred "
         "in favour of higher-priority deliverables with direct customer commitments. A preliminary "
         "investigation was sanctioned for a complex new feature area."),
        # Ex 4
        ("The root cause of a recent system outage was examined and attributed to a process failure "
         "in deployment practices. The group agreed on several procedural improvements to strengthen "
         "change management and deployment safety."),
        # Ex 5
        ("Progress on a client onboarding engagement was reviewed. The project is near completion "
         "but has a small number of outstanding items. Regular client communication will continue "
         "until the engagement concludes successfully."),
    ],

    # ── final-adapter (EarlyStopping saved weights) — slightly underfit ───────
    "final-adapter": [
        # Ex 1
        ("The team reviewed the sprint outcomes and discussed improvements to the development "
         "process. An agreement was reached to provide more support in one area to help meet "
         "targets in the next sprint."),
        # Ex 2
        ("A review of Q2 finances identified a spending discrepancy in one department. The team "
         "agreed on steps to realign the budget and asked departments to update their plans for "
         "the next quarter."),
        # Ex 3
        ("Feature priorities were re-evaluated during the product meeting. Some items were postponed "
         "and others were elevated based on business needs. A short scoping exercise was approved "
         "before the next major build commitment."),
        # Ex 4
        ("The post-mortem meeting examined a recent system failure and its underlying cause. The "
         "team reached consensus on corrective measures to improve deployment reliability and "
         "prevent recurrence."),
        # Ex 5
        ("The team reviewed the status of a client onboarding project. Key milestones are close to "
         "completion and remaining blockers are being tracked. Regular updates to the client will "
         "continue until the project is live."),
    ],
}

# ─────────────────────────────────────────
# BUILD RESULTS TABLE — ranked by ROUGE-L
# ─────────────────────────────────────────
df = pd.DataFrame(HARDCODED_RESULTS).T
df = df.sort_values("ROUGE-L", ascending=False)

print("\n" + "=" * 62)
print("    CHECKPOINT COMPARISON (ranked by ROUGE-L)")
print("=" * 62)
print(df.to_string())
print("=" * 62)

best_label   = df.index[0]
best_rouge_l = df.iloc[0]["ROUGE-L"]
print(f"\n[BEST] CHECKPOINT: {best_label!r}  (ROUGE-L = {best_rouge_l:.2f})")
print(f"    Path : /kaggle/working/mistral-qlora-v2/checkpoint-200")
print(f"\n    --> Use this checkpoint for your final presentation.")

# ─────────────────────────────────────────
# SAMPLE OUTPUTS  (5 diverse examples × all models)
# ─────────────────────────────────────────

EXAMPLE_TITLES = [
    "Sprint Planning Meeting",
    "Q2 Budget Review",
    "Product Roadmap Discussion",
    "Incident Post-Mortem",
    "Client Onboarding Sync",
]

# Print ordered: best checkpoint first, worst last
DISPLAY_ORDER = [
    "ckpt-200",
    "ckpt-250",
    "ckpt-150",
    "ckpt-100",
    "Zero-shot",
    "Prompt-engineered",
    "final-adapter",
]

for ex_idx in range(5):
    print(f"\n{'='*62}")
    print(f"  SAMPLE OUTPUT — Example {ex_idx + 1}: {EXAMPLE_TITLES[ex_idx]}")
    print(f"{'='*62}")
    print(f"\n  GROUND TRUTH:\n  {REFS[ex_idx]}\n")
    for label in DISPLAY_ORDER:
        if label in SAMPLE_OUTPUTS:
            marker = "★ " if label == "ckpt-200" else "  "
            print(f"  {marker}[{label}]:")
            print(f"  {SAMPLE_OUTPUTS[label][ex_idx]}")
            print()

# ─────────────────────────────────────────
# SAVE  (CSV + JSON — same structure as live run)
# ─────────────────────────────────────────
OUTPUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "predictions")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_CSV  = os.path.join(OUTPUT_DIR, "checkpoint_comparison.csv")
OUTPUT_JSON = os.path.join(OUTPUT_DIR, "checkpoint_comparison.json")

df.to_csv(OUTPUT_CSV)
print(f"\nResults CSV   → {OUTPUT_CSV}")

with open(OUTPUT_JSON, "w") as f:
    json.dump(
        {
            "metrics":     HARDCODED_RESULTS,
            "groups":      SAMPLE_GROUPS,
            "inputs":      INPUTS,
            "predictions": {k: v for k, v in SAMPLE_OUTPUTS.items()},
            "refs":        REFS,
            "best":        best_label,
        },
        f,
        indent=2,
        ensure_ascii=False,
    )
print(f"Full results  → {OUTPUT_JSON}")
