const $ = (id) => document.getElementById(id);
const pct = (x, digits=2) => x == null ? '—' : (x * 100).toFixed(digits) + '%';
const num = (x, digits=3) => x == null ? '—' : Number(x).toFixed(digits);
const money = (x) => '¥' + Number(x).toLocaleString('zh-CN', {maximumFractionDigits:0});
const escapeHtml = (s) => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let research;
let chartMode = 'wealth';

document.querySelectorAll('.tab').forEach(button => button.addEventListener('click', () => {
  document.querySelectorAll('.tab').forEach(tab => {
    const selected = tab === button;
    tab.classList.toggle('active', selected);
    tab.setAttribute('aria-selected', String(selected));
    $(tab.dataset.tab + '-panel').hidden = !selected;
  });
}));
document.querySelectorAll('[data-chart]').forEach(button => button.addEventListener('click', () => {
  chartMode = button.dataset.chart;
  document.querySelectorAll('[data-chart]').forEach(x => x.classList.toggle('selected', x === button));
  if (research) renderChart();
}));

function chartSeries() {
  const rows = research.daily;
  if (chartMode === 'wealth') return rows.map(r => [r[1], r[2]]);
  let pmax = 1, bmax = 1;
  return rows.map(r => {
    pmax = Math.max(pmax, r[1]); bmax = Math.max(bmax, r[2]);
    return [r[1] / pmax - 1, r[2] / bmax - 1];
  });
}
function renderChart() {
  const data = chartSeries(), n = data.length, W = 850, H = 320, left = 55, right = 17, top = 18, bottom = 35;
  const values = data.flat(), lo = chartMode === 'wealth' ? Math.min(.95, ...values) : Math.min(-.15, ...values);
  const hi = chartMode === 'wealth' ? Math.max(1.42, ...values) : .015;
  const x = i => left + (W-left-right)*i/(n-1), y = v => top + (H-top-bottom)*(hi-v)/(hi-lo);
  const line = j => data.map((r,i) => (i ? 'L':'M') + x(i).toFixed(2) + ',' + y(r[j]).toFixed(2)).join(' ');
  const ticks = Array.from({length:5},(_,i) => lo + (hi-lo)*i/4);
  const label = v => chartMode === 'wealth' ? v.toFixed(2) : (v*100).toFixed(0)+'%';
  const grid = ticks.map(v => '<line x1="'+left+'" y1="'+y(v)+'" x2="'+(W-right)+'" y2="'+y(v)+'" stroke="#31445a" stroke-width="1"/><text x="'+(left-10)+'" y="'+(y(v)+4)+'" text-anchor="end" fill="#8da1b5" font-size="11">'+label(v)+'</text>').join('');
  const years = [['2024',0],['2025',research.daily.findIndex(r=>r[0]>='2025-01-01')],['2026',research.daily.findIndex(r=>r[0]>='2026-01-01')]].map(([t,i]) => '<text x="'+x(i)+'" y="'+(H-7)+'" text-anchor="middle" fill="#8da1b5" font-size="11">'+t+'</text>').join('');
  $('portfolio-chart').innerHTML = '<svg viewBox="0 0 '+W+' '+H+'" role="img" aria-label="'+(chartMode==='wealth'?'策略与基准累计净值':'策略与基准历史回撤')+'"><title>'+ (chartMode==='wealth'?'累计净值':'历史回撤') +'</title>'+grid+years+'<path d="'+line(1)+'" fill="none" stroke="#e9bd72" stroke-width="2.5"/><path d="'+line(0)+'" fill="none" stroke="#70e1c0" stroke-width="3"/><rect id="chart-hit" x="'+left+'" y="'+top+'" width="'+(W-left-right)+'" height="'+(H-top-bottom)+'" fill="transparent"/></svg><div class="chart-tooltip" id="chart-tooltip" hidden></div>';
  const hit = $('chart-hit'), tip = $('chart-tooltip');
  hit.addEventListener('pointermove', ev => {
    const box = hit.ownerSVGElement.getBoundingClientRect();
    const frac = (ev.clientX - box.left) / box.width;
    const i = Math.max(0,Math.min(n-1,Math.round((frac-left/W)/((W-left-right)/W)*(n-1))));
    const row = research.daily[i];
    tip.textContent = row[0] + ' · 组合 ' + (chartMode==='wealth'?data[i][0].toFixed(3):pct(data[i][0])) + ' · 基准 ' + (chartMode==='wealth'?data[i][1].toFixed(3):pct(data[i][1]));
    tip.style.left = (ev.clientX - $('portfolio-chart').getBoundingClientRect().left) + 'px';
    tip.style.top = (ev.clientY - $('portfolio-chart').getBoundingClientRect().top) + 'px';
    tip.hidden = false;
  });
  hit.addEventListener('pointerleave', () => {tip.hidden = true});
}
function renderYearBars() {
  $('year-bars').innerHTML = research.yearly_relative.map(r => {
    const v=r.portfolio_log_relative, width=Math.min(100,Math.abs(v)/.18*100);
    return '<div class="year-row"><b>'+r.year+'</b><div class="year-track"><div class="year-fill '+(v<0?'negative':'')+'" style="width:'+width+'%"></div></div><span>'+ (v>=0?'+':'')+num(v,3)+'</span></div>';
  }).join('');
}
function renderDrivers() {
  const year=Number($('driver-year').value), direction=$('driver-direction').value;
  const rows=research.top_drivers.filter(r=>r.year===year && r.direction===direction).sort((a,b)=>a.rank-b.rank);
  $('drivers-list').innerHTML = rows.length ? rows.map(r=>'<div class="driver"><span>'+r.rank.toString().padStart(2,'0')+'</span><b>'+escapeHtml(r.instrument)+'</b><strong class="'+(r.net_pnl_cny<0?'negative':'')+'">'+money(r.net_pnl_cny)+'</strong></div>').join('') : '<p>该筛选条件没有已发布的前十名摘要。</p>';
}
function renderFactor() {
  const name=$('factor-select').value, f=research.factors.find(r=>r.name===name);
  if (!f) return;
  $('factor-metrics').innerHTML = [
    ['月均 Rank IC',num(f.mean_rank_ic,4)],['ICIR（未年化）',num(f.icir,3)],['BH q',num(f.bh_q,3)],
    ['HAC p',num(f.hac_p,3)],['有效月份',String(f.months)],['平均股票数',num(f.coverage_stocks,0)]
  ].map(([label,val])=>'<div><span>'+label+'</span><strong>'+val+'</strong></div>').join('');
  $('factor-spread').innerHTML = [5,10].map(n=>{
    const v=f['spread_'+n], width=v==null?0:Math.min(100,Math.abs(v)/.018*100);
    return '<div class="spread-row"><span>'+n+' 组高－低</span><div class="spread-track"><div class="spread-fill '+(v<0?'negative':'')+'" style="width:'+width+'%"></div></div><strong>'+pct(v)+'</strong></div>';
  }).join('');
}
function renderFactorTable() {
  $('factor-table').querySelector('tbody').innerHTML = research.factors.map(f=>
    '<tr><td>'+escapeHtml(f.name)+'</td><td>'+num(f.mean_rank_ic,4)+'</td><td>'+num(f.icir,3)+'</td><td>'+num(f.hac_p,3)+'</td><td>'+num(f.bh_q,3)+'</td><td>'+f.months+'</td></tr>'
  ).join('');
}
function renderHeatmap() {
  const {names,values}=research.correlation, root=$('corr-heatmap');
  const frag=document.createDocumentFragment();
  values.forEach((row,i)=>row.forEach((v,j)=>{
    const b=document.createElement('button');
    const strength=Math.max(0,Math.min(1,v));
    b.style.background='hsl('+(163-125*strength)+' 54% '+(22+40*strength)+'%)';
    b.title=names[i]+' × '+names[j]+': '+num(v,3);
    b.setAttribute('aria-label',b.title);
    b.addEventListener('click',()=>{$('corr-detail').textContent=b.title+' · 跨月平均绝对相关'});
    frag.appendChild(b);
  }));
  root.replaceChildren(frag);
}
function renderRegressions() {
  $('regression-cards').innerHTML = research.regressions.map(r=>'<div><span>'+ (r.model==='CH4'?'CH-4 四因子':'CH 市场单因子') +'</span><strong>'+pct(r.alpha_linear_annual)+'</strong><small>HAC p = '+num(r.alpha_hac_p,3)+' · R² = '+num(r.r_squared,3)+'</small></div>').join('');
}
function renderStress() {
  const cap=Number($('capital-select').value), rows=research.execution_stress.filter(r=>r.capital_cny===cap).sort((a,b)=>a.slippage_bps-b.slippage_bps);
  $('stress-bars').innerHTML = rows.map(r=>'<div class="stress-row"><span>滑点 '+num(r.slippage_bps,0)+' bp</span><div class="stress-track"><div class="stress-fill" style="width:'+Math.max(0,Math.min(100,r.total_return/.45*100))+'%"></div></div><strong>'+pct(r.total_return)+'</strong></div>').join('');
}
async function init() {
  try {
    const response=await fetch('data/research.json');
    if(!response.ok) throw new Error('HTTP '+response.status);
    research=await response.json();
    if(research.schema!=='equity-alpha-lab-showcase-v1' || research.daily.length!==667 || research.factors.length!==28) throw new Error('数据结构与冻结研究不符');
    renderChart(); renderYearBars(); renderDrivers();
    $('factor-select').innerHTML=research.factors.map(f=>'<option value="'+escapeHtml(f.name)+'">'+escapeHtml(f.name)+'</option>').join('');
    renderFactor(); renderFactorTable(); renderHeatmap(); renderRegressions(); renderStress();
    $('driver-year').addEventListener('change',renderDrivers);
    $('driver-direction').addEventListener('change',renderDrivers);
    $('factor-select').addEventListener('change',renderFactor);
    $('capital-select').addEventListener('change',renderStress);
  } catch(err) {
    $('portfolio-chart').textContent='研究数据加载失败：'+err.message+'。请通过 HTTP 服务打开本站，或检查 data/research.json。';
  }
}
init();

async function initCandidate() {
  try {
    const response=await fetch('data/candidate.json');
    if(!response.ok) throw new Error('HTTP '+response.status);
    const candidate=await response.json();
    const draw=()=>{
      const s=candidate.scenarios[$('candidate-scenario').value], rows=s.daily;
      const values=rows.flatMap(r=>[r[1],r[2]]), lo=Math.min(1,...values), hi=Math.max(...values);
      const x=i=>55+770*i/(rows.length-1), y=v=>20+260*(hi-v)/(hi-lo);
      const line=j=>rows.map((r,i)=>(i?'L':'M')+x(i).toFixed(2)+','+y(r[j]).toFixed(2)).join(' ');
      const grid=Array.from({length:5},(_,i)=>{const v=lo+(hi-lo)*i/4;return '<text x="8" y="'+(y(v)+4)+'" fill="#aabccd" font-size="12">'+v.toFixed(2)+'</text><line x1="55" x2="825" y1="'+y(v)+'" y2="'+y(v)+'" stroke="#31445a"/>';}).join('');
      $('candidate-chart').innerHTML='<svg viewBox="0 0 850 320" role="img" aria-label="趋势融合候选与价格指数净值"><title>TrendBlend20 与沪深300价格指数</title>'+grid+'<path d="'+line(2)+'" fill="none" stroke="#e9bd72" stroke-width="2"/><path d="'+line(1)+'" fill="none" stroke="#70e1c0" stroke-width="3"/><text x="55" y="310" fill="#aabccd">2024-01</text><text x="750" y="310" fill="#aabccd">2026-10</text></svg>';
      $('candidate-results').innerHTML='<div class="table-scroll"><table><thead><tr><th>年份</th><th>候选收益</th><th>基准收益</th><th>收益差（百分点）</th></tr></thead><tbody>'+s.annual.map(r=>'<tr><td>'+r.year+(r.year===2026?'（截至10月8日）':'')+'</td><td>'+pct(r.strategy_return)+'</td><td>'+pct(r.benchmark_return)+'</td><td>'+num(100*r.active_pp,2)+'</td></tr>').join('')+'</tbody></table></div><p>累计收益 <strong>'+pct(s.total_return)+'</strong> · 最大回撤 <strong>'+pct(s.max_drawdown)+'</strong> · 最高日末单股权重 '+pct(s.max_marked_weight)+'</p>';
    };
    $('candidate-scenario').addEventListener('change',draw);draw();
  } catch(err) { $('candidate-chart').textContent='候选数据加载失败：'+err.message; }
}
initCandidate();
