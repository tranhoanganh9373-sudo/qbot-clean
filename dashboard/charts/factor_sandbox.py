"""Dashboard panel — 因子沙盒 (Sandbox).

读 data_cache/sandbox_factors.json (全 universe z-scores), 嵌为 JS const,
3 个滑块 (z_pred / amp_imb_20d / JZF) 实时重算 final_score 并重排 Top 8.

严格 sandbox: production paper_trade_today.py 锁定 λ 不动, 本面板纯浏览器演算,
不写任何文件、不发任何请求.

数据源 (只读):
  - data_cache/sandbox_factors.json (paper_trade_today --dry-run 产出)
"""
from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SANDBOX_PATH = ROOT / "data_cache" / "sandbox_factors.json"


def _placeholder(msg: str) -> str:
    return (
        '<div class="placeholder-content" style="padding:24px 16px;">'
        f"{msg}"
        "</div>"
    )


def build_factor_sandbox_section() -> str:
    if not SANDBOX_PATH.exists():
        return _placeholder(
            "<code>data_cache/sandbox_factors.json</code> 不存在 — 跑 "
            "<code>paper_trade_today.py --dry-run</code> 后会自动产出."
        )
    try:
        data = json.loads(SANDBOX_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        return _placeholder(f"sandbox_factors.json 解析失败: {type(e).__name__}: {e}")

    as_of = data.get("as_of_date", "?")
    pv = data.get("production_version", "?")
    weights = data.get("production_weights", {})
    meta = data.get("factor_meta", {})
    universe = data.get("universe", [])
    prod_picks = data.get("production_picks", [])
    k = int(data.get("k", 8))
    n_universe = int(data.get("n_universe", len(universe)))
    generated_at = data.get("generated_at", "")

    w_pred_init = float(weights.get("z_pred", 1.0))
    w_amp_init = float(weights.get("amp_imb_20d", -0.30))
    w_jzf_init = float(weights.get("JZF", 0.10))

    z_pred_meta = meta.get("z_pred", {})
    amp_meta = meta.get("amp_imb_20d", {})
    jzf_meta = meta.get("JZF", {})

    universe_json = json.dumps(universe, ensure_ascii=False)
    prod_picks_json = json.dumps(prod_picks)

    banner = (
        '<div style="background:rgba(59,130,246,0.08); border-left:3px solid #3b82f6; '
        'padding:10px 14px; margin-bottom:12px; border-radius:0 4px 4px 0; font-size:12px;">'
        '<strong>🧪 因子沙盒</strong> · 拖动滑块查看不同权重下的 Top 8 picks · '
        f'production <strong>{html.escape(pv)}</strong> '
        f'as-of <strong>{html.escape(as_of)}</strong> · '
        f'universe <strong>{n_universe}</strong> 只 · '
        f'生成 <code>{html.escape(generated_at[:19])}</code>'
        '<br><span style="color:var(--muted, #6b7280); font-size:11px;">'
        '严格 sandbox: paper_trade_today.py λ 不动, 滑块仅浏览器内重排. '
        '点 <strong>"重置 production"</strong> 恢复 v19.10 锁定值. '
        '<strong>注意</strong>: production λ 通过严格 OOS 协议锁定, 用滑块"挑"出更优 λ '
        '= post-hoc 调参 (违反协议). 此面板用于教学和敏感性分析, 不用作改 production 依据.'
        '</span></div>'
    )

    def _slider(slug: str, label: str, init: float, sign: str, desc: str,
                vmin: float, vmax: float, step: float) -> str:
        sign_color = "#16a34a" if sign == "+" else "#dc2626"
        return (
            f'<div class="sb-slider" data-slug="{slug}" '
            f'style="margin-bottom:14px; padding:10px 12px; '
            f'background:rgba(107,114,128,0.06); border-left:3px solid {sign_color}; '
            f'border-radius:0 4px 4px 0;">'
            f'<div style="display:flex; justify-content:space-between; '
            f'align-items:center; margin-bottom:6px;">'
            f'<div>'
            f'<strong style="font-size:13px;">{html.escape(label)}</strong> '
            f'<span style="color:{sign_color}; font-size:11px; font-weight:600;">'
            f'sign {html.escape(sign)}</span>'
            f'<span style="color:var(--muted, #6b7280); font-size:11px; margin-left:8px;">'
            f'{html.escape(desc)}</span>'
            f'</div>'
            f'<div style="font-family:monospace; font-size:14px; font-weight:700;">'
            f'λ = <span id="sb-val-{slug}">{init:+.3f}</span>'
            f'</div>'
            f'</div>'
            f'<input type="range" id="sb-input-{slug}" '
            f'min="{vmin}" max="{vmax}" step="{step}" value="{init}" '
            f'style="width:100%;">'
            f'<div style="display:flex; justify-content:space-between; '
            f'font-size:10px; color:var(--muted, #6b7280); margin-top:2px;">'
            f'<span>{vmin:+.2f}</span>'
            f'<span style="color:#3b82f6;">production = {init:+.3f}</span>'
            f'<span>{vmax:+.2f}</span>'
            f'</div>'
            f'</div>'
        )

    sliders = (
        _slider("pred", "z(pred) Alpha158+DoubleEnsemble", w_pred_init, "+",
                z_pred_meta.get("desc", ""), 0.0, 2.0, 0.05)
        + _slider("amp", "z(amp_imb_20d) 振幅反转", w_amp_init, "-",
                  amp_meta.get("desc", ""), -1.0, 0.5, 0.05)
        + _slider("jzf", "z(JZF) 集合竞价跳空", w_jzf_init, "+",
                  jzf_meta.get("desc", ""), -0.5, 1.0, 0.05)
    )

    controls = (
        '<div style="display:flex; gap:8px; margin:8px 0 14px 0; flex-wrap:wrap;">'
        '<button id="sb-reset" style="padding:6px 14px; font-size:12px; '
        'background:#3b82f6; color:white; border:none; border-radius:4px; cursor:pointer;">'
        '↻ 重置 production'
        '</button>'
        '<button id="sb-zero" style="padding:6px 14px; font-size:12px; '
        'background:rgba(107,114,128,0.12); border:1px solid rgba(107,114,128,0.30); '
        'border-radius:4px; cursor:pointer;">'
        '🧹 全部归 0 (除 z_pred=1)'
        '</button>'
        '<button id="sb-only-amp" style="padding:6px 14px; font-size:12px; '
        'background:rgba(107,114,128,0.12); border:1px solid rgba(107,114,128,0.30); '
        'border-radius:4px; cursor:pointer;">'
        '试: 只 amp_imb'
        '</button>'
        '<button id="sb-only-jzf" style="padding:6px 14px; font-size:12px; '
        'background:rgba(107,114,128,0.12); border:1px solid rgba(107,114,128,0.30); '
        'border-radius:4px; cursor:pointer;">'
        '试: 只 JZF'
        '</button>'
        '<button id="sb-pred-only" style="padding:6px 14px; font-size:12px; '
        'background:rgba(107,114,128,0.12); border:1px solid rgba(107,114,128,0.30); '
        'border-radius:4px; cursor:pointer;">'
        '试: 纯 baseline'
        '</button>'
        '</div>'
    )

    table_shell = f"""
<h3 style="font-size:13px; margin:14px 0 6px 0; color:var(--muted, #6b7280);">
  📊 重算 Top {k} (按 final_score 倒序)
  <span id="sb-diff-banner" style="font-size:11px; color:#16a34a; margin-left:8px;"></span>
</h3>
<table class="data" id="sb-result-table">
  <colgroup>
    <col style="width:5%;"><col style="width:13%;"><col style="width:16%;">
    <col style="width:7%;"><col style="width:12%;">
    <col style="width:9%;"><col style="width:9%;"><col style="width:9%;">
    <col style="width:8%;"><col style="width:12%;">
  </colgroup>
  <thead>
    <tr>
      <th>#</th><th>代码</th><th>名称</th>
      <th>状态</th><th>final</th>
      <th>z_pred</th><th>z_amp</th><th>z_jzf</th>
      <th>价</th><th>vs production</th>
    </tr>
  </thead>
  <tbody id="sb-result-tbody"></tbody>
</table>
<div style="margin-top:8px; font-size:11px; color:var(--muted, #6b7280);">
  状态 ★ = production v19.10 选中 · ◆ = 滑块新晋 · △ = production 选中但被挤出 ·
  vs production = rank Δ (↑ N 升 N 位) · 全 universe N = <strong id="sb-n-universe">{n_universe}</strong>
</div>
<details style="margin-top:14px; font-size:12px; color:var(--muted, #6b7280);">
<summary style="cursor:pointer;">📋 全 universe 排名 (展开查看 {n_universe} 只)</summary>
<table class="data" id="sb-full-table" style="margin-top:6px; font-size:11px;">
  <thead><tr>
    <th>#</th><th>代码</th><th>名称</th><th>final</th>
    <th>z_pred</th><th>z_amp</th><th>z_jzf</th><th>价</th>
  </tr></thead>
  <tbody id="sb-full-tbody"></tbody>
</table>
</details>
"""

    script = f"""
<script>
(function() {{
  const SB_UNIVERSE = {universe_json};
  const SB_PROD_PICKS = {prod_picks_json};
  const SB_K = {k};
  const SB_DEFAULTS = {{
    pred: {w_pred_init},
    amp: {w_amp_init},
    jzf: {w_jzf_init},
  }};

  const inputs = {{
    pred: document.getElementById('sb-input-pred'),
    amp: document.getElementById('sb-input-amp'),
    jzf: document.getElementById('sb-input-jzf'),
  }};
  const vals = {{
    pred: document.getElementById('sb-val-pred'),
    amp: document.getElementById('sb-val-amp'),
    jzf: document.getElementById('sb-val-jzf'),
  }};
  const topBody = document.getElementById('sb-result-tbody');
  const fullBody = document.getElementById('sb-full-tbody');
  const diffBanner = document.getElementById('sb-diff-banner');

  function fmtSigned(v, digits) {{
    if (v == null || isNaN(v)) return '—';
    const s = v >= 0 ? '+' : '';
    return s + v.toFixed(digits == null ? 3 : digits);
  }}
  function fmtPrice(v) {{
    if (v == null || isNaN(v)) return '—';
    return '¥' + v.toFixed(2);
  }}
  function colorClass(v) {{
    if (v == null) return '';
    if (v > 0) return 'color:#16a34a;';
    if (v < 0) return 'color:#dc2626;';
    return '';
  }}

  function recompute() {{
    const wPred = parseFloat(inputs.pred.value);
    const wAmp = parseFloat(inputs.amp.value);
    const wJzf = parseFloat(inputs.jzf.value);
    vals.pred.textContent = fmtSigned(wPred);
    vals.amp.textContent = fmtSigned(wAmp);
    vals.jzf.textContent = fmtSigned(wJzf);

    const ranked = SB_UNIVERSE.map(r => {{
      const f = wPred * (r.z_pred || 0)
              + wAmp * (r.z_amp || 0)
              + wJzf * (r.z_jzf || 0);
      return Object.assign({{}}, r, {{ final_score: f }});
    }}).sort((a, b) => b.final_score - a.final_score);

    const sandboxTopSyms = ranked.slice(0, SB_K).map(r => r.sym);
    const prodRankMap = {{}};
    SB_PROD_PICKS.forEach((sym, i) => {{ prodRankMap[sym] = i + 1; }});

    let topHtml = '';
    ranked.slice(0, SB_K).forEach((r, i) => {{
      const isProd = SB_PROD_PICKS.indexOf(r.sym) >= 0;
      const status = isProd ? '<span style="color:#3b82f6;">★</span>'
                            : '<span style="color:#f59e0b;">◆</span>';
      const prodRank = prodRankMap[r.sym];
      let deltaCell;
      if (prodRank != null) {{
        const delta = prodRank - (i + 1);
        if (delta === 0) {{
          deltaCell = '<span style="color:var(--muted, #6b7280);">—</span>';
        }} else if (delta > 0) {{
          deltaCell = '<span style="color:#16a34a;">↑ ' + delta + '</span>';
        }} else {{
          deltaCell = '<span style="color:#dc2626;">↓ ' + (-delta) + '</span>';
        }}
      }} else {{
        deltaCell = '<span style="color:#f59e0b;">new</span>';
      }}
      const f = r.final_score;
      topHtml += '<tr>'
              + '<td>' + (i + 1) + '</td>'
              + '<td><code>' + r.sym + '</code></td>'
              + '<td>' + (r.name || '?') + '</td>'
              + '<td style="text-align:center;">' + status + '</td>'
              + '<td style="text-align:right; font-family:monospace; '
              + colorClass(f) + ' font-weight:600;">'
              + fmtSigned(f) + '</td>'
              + '<td style="text-align:right; font-family:monospace; '
              + colorClass(r.z_pred) + '">' + fmtSigned(r.z_pred, 2) + '</td>'
              + '<td style="text-align:right; font-family:monospace; '
              + colorClass(r.z_amp) + '">' + fmtSigned(r.z_amp, 2) + '</td>'
              + '<td style="text-align:right; font-family:monospace; '
              + colorClass(r.z_jzf) + '">' + fmtSigned(r.z_jzf, 2) + '</td>'
              + '<td style="text-align:right;">' + fmtPrice(r.close) + '</td>'
              + '<td style="text-align:right;">' + deltaCell + '</td>'
              + '</tr>';
    }});
    topBody.innerHTML = topHtml;

    const droppedFromProd = SB_PROD_PICKS.filter(s => sandboxTopSyms.indexOf(s) < 0);
    let droppedHtml = '';
    droppedFromProd.forEach(sym => {{
      const r = ranked.find(x => x.sym === sym);
      if (!r) return;
      const sandboxRank = ranked.indexOf(r) + 1;
      const prodRank = prodRankMap[sym];
      droppedHtml += '<tr style="opacity:0.55; background:rgba(220,38,38,0.05);">'
              + '<td>' + sandboxRank + '</td>'
              + '<td><code>' + r.sym + '</code></td>'
              + '<td>' + (r.name || '?') + '</td>'
              + '<td style="text-align:center; color:#dc2626;">△</td>'
              + '<td style="text-align:right; font-family:monospace; '
              + colorClass(r.final_score) + '">'
              + fmtSigned(r.final_score) + '</td>'
              + '<td style="text-align:right; font-family:monospace; '
              + colorClass(r.z_pred) + '">' + fmtSigned(r.z_pred, 2) + '</td>'
              + '<td style="text-align:right; font-family:monospace; '
              + colorClass(r.z_amp) + '">' + fmtSigned(r.z_amp, 2) + '</td>'
              + '<td style="text-align:right; font-family:monospace; '
              + colorClass(r.z_jzf) + '">' + fmtSigned(r.z_jzf, 2) + '</td>'
              + '<td style="text-align:right;">' + fmtPrice(r.close) + '</td>'
              + '<td style="text-align:right; color:#dc2626;">↓ ' + (sandboxRank - prodRank) + '</td>'
              + '</tr>';
    }});
    if (droppedHtml) topBody.innerHTML += droppedHtml;

    const overlapCount = sandboxTopSyms.filter(s => SB_PROD_PICKS.indexOf(s) >= 0).length;
    if (overlapCount === SB_K) {{
      diffBanner.textContent = '✓ 跟 production picks 完全一致';
      diffBanner.style.color = '#16a34a';
    }} else {{
      diffBanner.textContent = '⚠ 与 production 差异 ' + (SB_K - overlapCount) + ' 只 (重叠 '
                            + overlapCount + '/' + SB_K + ')';
      diffBanner.style.color = '#f59e0b';
    }}

    let fullHtml = '';
    ranked.forEach((r, i) => {{
      const isTop = i < SB_K;
      const bg = isTop ? 'background:rgba(59,130,246,0.05);' : '';
      fullHtml += '<tr style="' + bg + '">'
              + '<td>' + (i + 1) + '</td>'
              + '<td><code>' + r.sym + '</code></td>'
              + '<td>' + (r.name || '?') + '</td>'
              + '<td style="text-align:right; font-family:monospace; '
              + colorClass(r.final_score) + '">' + fmtSigned(r.final_score) + '</td>'
              + '<td style="text-align:right; font-family:monospace;">'
              + fmtSigned(r.z_pred, 2) + '</td>'
              + '<td style="text-align:right; font-family:monospace;">'
              + fmtSigned(r.z_amp, 2) + '</td>'
              + '<td style="text-align:right; font-family:monospace;">'
              + fmtSigned(r.z_jzf, 2) + '</td>'
              + '<td style="text-align:right;">' + fmtPrice(r.close) + '</td>'
              + '</tr>';
    }});
    fullBody.innerHTML = fullHtml;
  }}

  Object.keys(inputs).forEach(k => {{
    inputs[k].addEventListener('input', recompute);
  }});

  document.getElementById('sb-reset').addEventListener('click', () => {{
    inputs.pred.value = SB_DEFAULTS.pred;
    inputs.amp.value = SB_DEFAULTS.amp;
    inputs.jzf.value = SB_DEFAULTS.jzf;
    recompute();
  }});
  document.getElementById('sb-zero').addEventListener('click', () => {{
    inputs.pred.value = 1.0;
    inputs.amp.value = 0;
    inputs.jzf.value = 0;
    recompute();
  }});
  document.getElementById('sb-only-amp').addEventListener('click', () => {{
    inputs.pred.value = 0;
    inputs.amp.value = -1.0;
    inputs.jzf.value = 0;
    recompute();
  }});
  document.getElementById('sb-only-jzf').addEventListener('click', () => {{
    inputs.pred.value = 0;
    inputs.amp.value = 0;
    inputs.jzf.value = 1.0;
    recompute();
  }});
  document.getElementById('sb-pred-only').addEventListener('click', () => {{
    inputs.pred.value = 1.0;
    inputs.amp.value = 0;
    inputs.jzf.value = 0;
    recompute();
  }});

  recompute();
}})();
</script>
"""

    footer = (
        '<div style="margin-top:14px; font-size:11px; color:var(--muted, #6b7280);">'
        '<strong>公式</strong>: final_score = '
        'λ_pred × z(pred) + λ_amp × z(amp_imb_20d) + λ_jzf × z(JZF)<br>'
        '<strong>production v19.10 锁定值</strong>: '
        'λ_pred=+1.00 (baseline), λ_amp=−0.30 (Phase 4 IS sweep), '
        'λ_jzf=+0.10 (Phase A IS sweep)<br>'
        '<strong>OOS Calmar</strong>: '
        'baseline 0.77 · v19.6 (amp only) 1.29 · v19.10 (stacked) 2.12<br>'
        '<strong>数据源</strong>: <code>data_cache/sandbox_factors.json</code> · '
        'JS 浏览器内 sort · 不发任何请求'
        '</div>'
    )

    return banner + sliders + controls + table_shell + script + footer
