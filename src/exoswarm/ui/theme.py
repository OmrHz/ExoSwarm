"""Visual system for the ExoSwarm mission-control surface."""

MISSION_CONTROL_CSS = r"""
<style>
:root {
  --exo-bg: #04070d;
  --exo-panel: rgba(9, 17, 29, 0.84);
  --exo-panel-strong: #0b1422;
  --exo-line: rgba(117, 157, 197, 0.16);
  --exo-cyan: #41d9ff;
  --exo-teal: #49f2c2;
  --exo-amber: #ffbe55;
  --exo-red: #ff6577;
  --exo-purple: #a58bff;
  --exo-ink: #e9f2ff;
  --exo-muted: #8190a9;
}

.stApp {
  color: var(--exo-ink);
  background:
    radial-gradient(circle at 76% -8%, rgba(65, 217, 255, .10), transparent 33rem),
    radial-gradient(circle at 8% 48%, rgba(165, 139, 255, .07), transparent 27rem),
    linear-gradient(180deg, #060a11 0%, var(--exo-bg) 68%);
}

.stApp::before {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  opacity: .20;
  background-image:
    linear-gradient(rgba(110, 148, 185, .055) 1px, transparent 1px),
    linear-gradient(90deg, rgba(110, 148, 185, .055) 1px, transparent 1px);
  background-size: 42px 42px;
  mask-image: linear-gradient(to bottom, black, transparent 82%);
}

[data-testid="stHeader"] { background: rgba(4, 7, 13, .65); }
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #08111d 0%, #060b13 100%);
  border-right: 1px solid var(--exo-line);
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #aebbd0; }
[data-testid="stMainBlockContainer"] { max-width: 1500px; padding-top: 2.4rem; }

h1, h2, h3 { letter-spacing: -.025em; color: var(--exo-ink); }
p { color: #aebbd0; }
code { color: var(--exo-cyan) !important; }

.exo-brand {
  display: flex;
  align-items: center;
  gap: .9rem;
  margin: .2rem 0 1.45rem;
}
.exo-mark {
  position: relative;
  width: 42px;
  height: 42px;
  border: 1px solid rgba(65,217,255,.45);
  border-radius: 50%;
  box-shadow: inset 0 0 18px rgba(65,217,255,.10), 0 0 24px rgba(65,217,255,.08);
}
.exo-mark::before {
  content: "";
  position: absolute;
  inset: 8px;
  border: 1px solid rgba(73,242,194,.7);
  border-radius: 50%;
  transform: rotate(-28deg) scaleY(.42);
}
.exo-mark::after {
  content: "";
  position: absolute;
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--exo-cyan);
  top: 5px;
  left: 17px;
  box-shadow: 0 0 12px var(--exo-cyan);
}
.exo-brand-name { font-weight: 760; letter-spacing: .08em; font-size: 1.05rem; }
.exo-brand-sub { color: var(--exo-muted); font-size: .68rem; letter-spacing: .13em; text-transform: uppercase; }

.exo-hero {
  display: flex;
  justify-content: space-between;
  align-items: flex-end;
  gap: 1.5rem;
  padding: 1.25rem 1.35rem 1.15rem;
  border: 1px solid var(--exo-line);
  border-radius: 16px;
  background: linear-gradient(115deg, rgba(10,22,37,.94), rgba(7,13,23,.72));
  box-shadow: 0 24px 70px rgba(0,0,0,.20), inset 0 1px rgba(255,255,255,.025);
  overflow: hidden;
  position: relative;
}
.exo-hero::after {
  content: "";
  position: absolute;
  right: -12%;
  top: -180%;
  width: 46%;
  height: 430%;
  background: linear-gradient(90deg, transparent, rgba(65,217,255,.045), transparent);
  transform: rotate(18deg);
  pointer-events: none;
}
.exo-eyebrow { color: var(--exo-cyan); font-size: .68rem; letter-spacing: .16em; text-transform: uppercase; font-weight: 700; }
.exo-target { margin: .2rem 0 0; font-size: clamp(1.65rem, 3vw, 2.55rem); font-weight: 700; letter-spacing: -.035em; }
.exo-trace { color: var(--exo-muted); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .72rem; margin-top: .35rem; }
.exo-statuses { display: flex; gap: .5rem; flex-wrap: wrap; justify-content: flex-end; }
.exo-chip {
  border: 1px solid var(--exo-line);
  background: rgba(2,7,13,.55);
  border-radius: 999px;
  padding: .48rem .72rem;
  color: #a9b8cd;
  font-size: .67rem;
  font-weight: 700;
  letter-spacing: .08em;
  text-transform: uppercase;
  white-space: nowrap;
}
.exo-chip.good { color: var(--exo-teal); border-color: rgba(73,242,194,.32); }
.exo-chip.live { color: var(--exo-cyan); border-color: rgba(65,217,255,.32); }
.exo-chip.warn { color: var(--exo-amber); border-color: rgba(255,190,85,.32); }
.exo-chip.bad { color: var(--exo-red); border-color: rgba(255,101,119,.35); }
.exo-chip.sealed::before { content: "◆"; margin-right: .42rem; color: var(--exo-amber); }

.exo-section { margin: 2.05rem 0 .7rem; }
.exo-section-kicker { color: var(--exo-cyan); font-size: .64rem; font-weight: 760; letter-spacing: .18em; text-transform: uppercase; }
.exo-section-title { color: var(--exo-ink); font-size: 1.28rem; font-weight: 680; margin-top: .15rem; }
.exo-section-copy { color: var(--exo-muted); max-width: 800px; font-size: .82rem; margin-top: .18rem; line-height: 1.5; }

.exo-workflow {
  display: grid;
  grid-template-columns: repeat(8, minmax(72px, 1fr));
  gap: .42rem;
  margin: .85rem 0 .15rem;
}
.exo-step {
  min-height: 67px;
  padding: .62rem .64rem;
  border-radius: 10px;
  border: 1px solid var(--exo-line);
  background: rgba(8,16,28,.64);
  color: #6f7e95;
  font-size: .64rem;
  line-height: 1.25;
  text-transform: uppercase;
  letter-spacing: .055em;
}
.exo-step-index { display: block; font-family: ui-monospace, monospace; font-size: .61rem; margin-bottom: .35rem; color: #526077; }
.exo-step.done { color: #bceee0; border-color: rgba(73,242,194,.22); background: rgba(26,86,72,.10); }
.exo-step.done .exo-step-index { color: var(--exo-teal); }
.exo-step.active { color: #d8f6ff; border-color: rgba(65,217,255,.44); background: rgba(35,125,151,.12); box-shadow: inset 0 0 24px rgba(65,217,255,.04); }
.exo-step.active .exo-step-index { color: var(--exo-cyan); }

.exo-metric {
  height: 100%;
  min-height: 132px;
  border-radius: 13px;
  border: 1px solid var(--exo-line);
  background: linear-gradient(145deg, rgba(11,22,37,.90), rgba(7,13,23,.73));
  padding: .86rem .92rem;
}
.exo-metric-label { color: var(--exo-muted); font-size: .65rem; letter-spacing: .10em; text-transform: uppercase; font-weight: 720; }
.exo-metric-value { color: var(--exo-ink); font-size: 1.43rem; line-height: 1.15; font-weight: 690; margin-top: .45rem; font-variant-numeric: tabular-nums; }
.exo-metric-unit { color: var(--exo-cyan); font-size: .67rem; margin-left: .28rem; font-weight: 650; }
.exo-metric-detail { color: #8f9cb1; font-size: .66rem; margin-top: .48rem; line-height: 1.35; }
.exo-provenance { color: #5f7089; font-family: ui-monospace, monospace; font-size: .58rem; margin-top: .42rem; overflow-wrap: anywhere; }

.exo-panel {
  border: 1px solid var(--exo-line);
  background: var(--exo-panel);
  border-radius: 13px;
  padding: .95rem 1rem;
  min-height: 118px;
}
.exo-panel-label { color: var(--exo-muted); text-transform: uppercase; letter-spacing: .11em; font-size: .62rem; font-weight: 730; }
.exo-panel-title { color: var(--exo-ink); font-size: 1rem; font-weight: 670; margin-top: .4rem; }
.exo-panel-copy { color: #98a8bd; font-size: .76rem; line-height: 1.48; margin-top: .45rem; }
.exo-panel-code { color: var(--exo-cyan); font-family: ui-monospace, monospace; font-size: .62rem; margin-top: .48rem; }
.exo-verdict { font-size: 1.2rem; font-weight: 780; letter-spacing: .08em; }
.exo-verdict.APPROVE { color: var(--exo-teal); }
.exo-verdict.REVISE { color: var(--exo-amber); }
.exo-verdict.VETO { color: var(--exo-red); }

.exo-hypothesis-row {
  display: grid;
  grid-template-columns: minmax(155px, 1fr) auto auto;
  align-items: center;
  gap: .7rem;
  padding: .56rem .7rem;
  border-bottom: 1px solid rgba(117,157,197,.10);
}
.exo-hypothesis-row:last-child { border-bottom: 0; }
.exo-h-name { color: #c7d4e7; font-size: .76rem; }
.exo-h-state { color: var(--exo-muted); font-size: .61rem; letter-spacing: .06em; }
.exo-h-weight { color: var(--exo-cyan); font-family: ui-monospace, monospace; font-size: .69rem; min-width: 3rem; text-align: right; }

.exo-evidence {
  position: relative;
  border: 1px solid var(--exo-line);
  border-left: 3px solid var(--exo-cyan);
  border-radius: 9px;
  background: rgba(8,16,28,.72);
  padding: .74rem .86rem .72rem;
  margin-bottom: .48rem;
}
.exo-evidence.positive { border-left-color: var(--exo-teal); }
.exo-evidence.warning { border-left-color: var(--exo-amber); }
.exo-evidence.negative { border-left-color: var(--exo-red); }
.exo-evidence-head { display: flex; justify-content: space-between; gap: 1rem; }
.exo-evidence-title { color: #d8e4f4; font-size: .78rem; font-weight: 650; }
.exo-evidence-id { color: #5e7088; font-family: ui-monospace, monospace; font-size: .59rem; }
.exo-evidence-meta { color: var(--exo-muted); font-size: .65rem; margin-top: .35rem; }

.exo-lock {
  position: relative;
  border: 1px solid rgba(73,242,194,.34);
  border-radius: 16px;
  padding: 1.3rem 1.35rem;
  overflow: hidden;
  background:
    radial-gradient(circle at 8% 50%, rgba(73,242,194,.10), transparent 22rem),
    linear-gradient(125deg, rgba(9,29,31,.80), rgba(7,13,23,.92));
  box-shadow: inset 0 0 42px rgba(73,242,194,.025), 0 18px 50px rgba(0,0,0,.18);
}
.exo-lock::after {
  content: "RESULT LOCKED";
  position: absolute;
  right: -1.1rem;
  top: 1.5rem;
  transform: rotate(8deg);
  border: 2px solid rgba(73,242,194,.16);
  color: rgba(73,242,194,.16);
  padding: .45rem 1.25rem;
  font-size: 1.3rem;
  font-weight: 820;
  letter-spacing: .12em;
}
.exo-lock-title { color: var(--exo-teal); font-size: 1.2rem; font-weight: 770; letter-spacing: .05em; }
.exo-lock-copy { color: #9fbdba; font-size: .77rem; margin-top: .35rem; max-width: 700px; }
.exo-hash { margin-top: .75rem; color: #b8fff0; font-family: ui-monospace, monospace; font-size: .67rem; overflow-wrap: anywhere; }

.exo-sealed {
  border: 1px dashed rgba(255,190,85,.32);
  border-radius: 13px;
  padding: 1.05rem 1.1rem;
  background: repeating-linear-gradient(-45deg, rgba(255,190,85,.025), rgba(255,190,85,.025) 8px, transparent 8px, transparent 16px);
}
.exo-sealed-title { color: var(--exo-amber); font-size: .8rem; font-weight: 760; letter-spacing: .10em; text-transform: uppercase; }
.exo-sealed-copy { color: #a79b83; font-size: .74rem; margin-top: .4rem; max-width: 760px; }

.exo-reveal {
  border: 1px solid rgba(165,139,255,.34);
  border-radius: 16px;
  padding: 1.2rem 1.3rem;
  background: radial-gradient(circle at 100% 0, rgba(165,139,255,.14), transparent 24rem), rgba(10,15,28,.88);
}
.exo-reveal-kicker { color: var(--exo-purple); text-transform: uppercase; letter-spacing: .15em; font-weight: 760; font-size: .64rem; }
.exo-reveal-name { color: #f1edff; font-size: 1.52rem; font-weight: 720; margin-top: .22rem; }
.exo-reveal-status { color: #bdaeff; font-size: .76rem; margin-top: .18rem; }

.exo-empty {
  text-align: center;
  padding: 5.5rem 1.5rem;
  border: 1px dashed rgba(65,217,255,.22);
  border-radius: 16px;
  background: rgba(7,14,24,.60);
}
.exo-empty-orbit { color: var(--exo-cyan); font-size: 2.2rem; opacity: .7; }
.exo-empty-title { color: var(--exo-ink); font-weight: 680; font-size: 1.12rem; margin-top: .5rem; }
.exo-empty-copy { color: var(--exo-muted); font-size: .78rem; max-width: 540px; margin: .45rem auto; line-height: 1.55; }

[data-testid="stMetric"], [data-testid="stDataFrame"], [data-testid="stPlotlyChart"] {
  border-radius: 12px;
}
[data-testid="stPlotlyChart"] { border: 1px solid rgba(117,157,197,.13); background: rgba(7,13,23,.48); }
[data-testid="stExpander"] { border-color: var(--exo-line); background: rgba(7,13,23,.52); }
.stTabs [data-baseweb="tab-list"] { gap: .3rem; border-bottom: 1px solid var(--exo-line); }
.stTabs [data-baseweb="tab"] { color: var(--exo-muted); font-size: .73rem; letter-spacing: .03em; }
.stTabs [aria-selected="true"] { color: var(--exo-cyan) !important; }

@media (max-width: 900px) {
  .exo-workflow { grid-template-columns: repeat(4, 1fr); }
  .exo-hero { align-items: flex-start; flex-direction: column; }
  .exo-statuses { justify-content: flex-start; }
}
@media (max-width: 580px) {
  .exo-workflow { grid-template-columns: repeat(2, 1fr); }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation-duration: .001ms !important; animation-iteration-count: 1 !important; }
}
</style>
"""

PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}

__all__ = ["MISSION_CONTROL_CSS", "PLOTLY_CONFIG"]
