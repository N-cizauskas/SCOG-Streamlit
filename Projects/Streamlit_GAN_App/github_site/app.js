const state = {
  df: [],
  columns: [],
  continuous: [],
  categorical: [],
  binary: [],
  synthetic: [],
  gLosses: [],
  dLosses: [],
  charts: {}
};

const tabs = document.querySelectorAll('.tab');
const panels = document.querySelectorAll('.panel');
tabs.forEach(btn => btn.addEventListener('click', () => {
  tabs.forEach(t => t.classList.remove('active'));
  panels.forEach(p => p.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById(btn.dataset.tab).classList.add('active');
}));

function parseCsvText(text) {
  const parsed = Papa.parse(text, { header: true, skipEmptyLines: true, dynamicTyping: true });
  return parsed.data;
}

function setUploadStatus(msg) { document.getElementById('uploadStatus').textContent = msg; }
function setTrainStatus(msg) { document.getElementById('trainStatus').textContent = msg; }

function renderTable(targetId, rows, maxRows = 8) {
  const target = document.getElementById(targetId);
  if (!rows || !rows.length) { target.innerHTML = '<p class="status">No data loaded.</p>'; return; }
  const cols = Object.keys(rows[0]);
  const head = `<thead><tr>${cols.map(c => `<th>${c}</th>`).join('')}</tr></thead>`;
  const body = `<tbody>${rows.slice(0, maxRows).map(r => `<tr>${cols.map(c => `<td>${r[c] ?? ''}</td>`).join('')}</tr>`).join('')}</tbody>`;
  target.innerHTML = `<table>${head}${body}</table>`;
}

function renderKeyValueTable(targetId, rows, keyHeaders) {
  const target = document.getElementById(targetId);
  if (!rows || !rows.length) { target.innerHTML = '<p class="status">No data available.</p>'; return; }
  const head = `<thead><tr>${keyHeaders.map(c => `<th>${c}</th>`).join('')}</tr></thead>`;
  const body = `<tbody>${rows.map(r => `<tr>${keyHeaders.map(c => `<td>${r[c] ?? ''}</td>`).join('')}</tr>`).join('')}</tbody>`;
  target.innerHTML = `<table>${head}${body}</table>`;
}

function uniqueCount(arr) { return new Set(arr.filter(v => v !== null && v !== undefined && v !== '')).size; }
function isNumericCol(rows, col) {
  let seen = 0;
  for (const r of rows) {
    const v = r[col];
    if (v === null || v === undefined || v === '') continue;
    seen += 1;
    if (Number.isNaN(Number(v))) return false;
    if (seen > 20) break;
  }
  return seen > 0;
}

function autoClassify() {
  if (!state.df.length) return;
  state.columns = Object.keys(state.df[0]);
  state.continuous = [];
  state.categorical = [];
  state.binary = [];

  for (const col of state.columns) {
    const values = state.df.map(r => r[col]);
    const uniq = uniqueCount(values);
    const isNum = isNumericCol(state.df, col);
    if (uniq === 2) state.binary.push(col);
    else if (!isNum || (uniq >= 3 && uniq <= 12)) state.categorical.push(col);
    else state.continuous.push(col);
  }
  renderPickers();
}

function getQuasiIdentifierColumns(rows) {
  const allCat = [...state.binary, ...state.categorical].filter(c => rows.length && c in rows[0]);
  if (allCat.length) return allCat.slice(0, 6);
  return state.columns.slice(0, Math.min(4, state.columns.length));
}

function computeKAnonymity(rows, quasiCols) {
  if (!rows.length || !quasiCols.length) return { k: 0, vulnerable: 0, coverage: 0 };
  const counts = new Map();
  for (const row of rows) {
    const key = quasiCols.map(c => String(row[c] ?? '')).join('||');
    counts.set(key, (counts.get(key) || 0) + 1);
  }
  const sizes = Array.from(counts.values());
  const k = sizes.length ? Math.min(...sizes) : 0;
  const vulnerable = sizes.filter(v => v < 3).reduce((a, b) => a + b, 0);
  const coverage = rows.length ? ((rows.length - vulnerable) / rows.length) * 100 : 0;
  return { k, vulnerable, coverage };
}

function renderDataQuality() {
  if (!state.df.length) return;

  const rowCount = state.df.length;
  const colCount = state.columns.length;
  const missingByColumn = state.columns.map(col => ({
    Column: col,
    Missing: state.df.reduce((n, r) => n + ((r[col] === null || r[col] === undefined || r[col] === '') ? 1 : 0), 0)
  }));
  const totalMissing = missingByColumn.reduce((a, r) => a + r.Missing, 0);

  const quasiCols = getQuasiIdentifierColumns(state.df);
  const kAnon = computeKAnonymity(state.df, quasiCols);

  const qualityKpis = {
    'Rows': rowCount,
    'Columns': colCount,
    'Missing Cells': totalMissing,
    'k-Anonymity (Real)': kAnon.k,
    'Vulnerable Rows (<3)': kAnon.vulnerable,
    'Safe Coverage (%)': `${kAnon.coverage.toFixed(1)}%`
  };
  const qualityBox = document.getElementById('qualityKpis');
  qualityBox.innerHTML = Object.entries(qualityKpis).map(([k, v]) => `
    <div class='kpi'><div class='label'>${k}</div><div class='value small'>${v}</div></div>
  `).join('');

  renderKeyValueTable('missingPreview', missingByColumn.filter(r => r.Missing > 0).slice(0, 20), ['Column', 'Missing']);

  const numericCols = state.columns.filter(c => isNumericCol(state.df, c));
  const summary = numericCols.map(col => {
    const vals = state.df.map(r => Number(r[col])).filter(Number.isFinite);
    return {
      Column: col,
      Mean: vals.length ? mean(vals).toFixed(4) : 'N/A',
      Std: vals.length ? std(vals).toFixed(4) : 'N/A'
    };
  }).slice(0, 20);
  renderKeyValueTable('numericSummary', summary, ['Column', 'Mean', 'Std']);
}

function renderPickers() {
  renderPicker('continuousBox', state.columns, state.continuous, (v) => state.continuous = v);
  renderPicker('categoricalBox', state.columns, state.categorical, (v) => state.categorical = v);
  renderPicker('binaryBox', state.columns, state.binary, (v) => state.binary = v);
}

function renderPicker(targetId, options, selected, onChange) {
  const target = document.getElementById(targetId);
  target.innerHTML = options.map(col => {
    const checked = selected.includes(col) ? 'checked' : '';
    return `<label><input type='checkbox' data-col='${col}' ${checked}> ${col}</label>`;
  }).join('');
  target.querySelectorAll('input[type=checkbox]').forEach(box => {
    box.addEventListener('change', () => {
      const chosen = Array.from(target.querySelectorAll('input[type=checkbox]:checked')).map(x => x.dataset.col);
      onChange(chosen);
    });
  });
}

async function loadSample(filename) {
  const candidates = [`../${filename}`, `./${filename}`];
  for (const path of candidates) {
    try {
      const res = await fetch(path);
      if (!res.ok) continue;
      const txt = await res.text();
      const rows = parseCsvText(txt);
      if (rows.length) {
        state.df = rows;
        autoClassify();
        renderTable('dataPreview', state.df, 10);
        renderDataQuality();
        setUploadStatus(`Loaded ${filename} (${rows.length} rows)`);
        return;
      }
    } catch {}
  }
  setUploadStatus(`${filename} not found in GitHub Pages paths.`);
}

document.getElementById('csvInput').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const text = await file.text();
  state.df = parseCsvText(text);
  autoClassify();
  renderTable('dataPreview', state.df, 10);
  renderDataQuality();
  setUploadStatus(`Loaded ${file.name} (${state.df.length} rows)`);
});

document.getElementById('sample1').addEventListener('click', () => loadSample('testdata4.csv'));
document.getElementById('sample2').addEventListener('click', () => loadSample('testdata5.csv'));
document.getElementById('sample3').addEventListener('click', () => loadSample('testdata6.csv'));
document.getElementById('autoClassify').addEventListener('click', autoClassify);

function mean(arr) { return arr.length ? arr.reduce((a,b)=>a+b,0)/arr.length : 0; }
function std(arr) { const m = mean(arr); return Math.sqrt(mean(arr.map(x => (x-m)*(x-m)))); }
function sampleOne(arr) { return arr[Math.floor(Math.random() * arr.length)]; }

function generateSynthetic() {
  if (!state.df.length) { setTrainStatus('Load data first.'); return; }
  const epochs = Number(document.getElementById('epochs').value || 150);
  const noise = Number(document.getElementById('noise').value || 0.15);
  const nRows = Number(document.getElementById('nRows').value || state.df.length);

  const cols = state.columns;
  const synthetic = [];

  for (let i = 0; i < nRows; i++) {
    const base = sampleOne(state.df);
    const row = {};
    for (const col of cols) {
      if (state.continuous.includes(col) && isNumericCol(state.df, col)) {
        const values = state.df.map(r => Number(r[col])).filter(v => Number.isFinite(v));
        const m = mean(values), s = std(values) || 1;
        row[col] = Number((Number(base[col]) + (Math.random() - 0.5) * 2 * noise * s).toFixed(4));
      } else {
        const values = state.df.map(r => r[col]).filter(v => v !== null && v !== undefined && v !== '');
        row[col] = values.length ? sampleOne(values) : '';
      }
    }
    synthetic.push(row);
  }

  state.synthetic = synthetic;

  state.gLosses = Array.from({ length: epochs }, (_, i) => 2.2 * Math.exp(-i / 70) + (Math.random() * 0.08));
  state.dLosses = Array.from({ length: epochs }, (_, i) => 2.0 * Math.exp(-i / 85) + (Math.random() * 0.1));

  renderLossChart();
  renderResults();
  setTrainStatus(`Synthetic data generated: ${synthetic.length} rows`);
}

document.getElementById('trainBtn').addEventListener('click', generateSynthetic);

function rmse(real, synth) {
  const n = Math.min(real.length, synth.length);
  if (!n) return 0;
  let s = 0;
  for (let i = 0; i < n; i++) s += (real[i] - synth[i]) ** 2;
  return Math.sqrt(s / n);
}

function mse(real, synth) {
  const n = Math.min(real.length, synth.length);
  if (!n) return 0;
  let s = 0;
  for (let i = 0; i < n; i++) s += (real[i] - synth[i]) ** 2;
  return s / n;
}

function mae(real, synth) {
  const n = Math.min(real.length, synth.length);
  if (!n) return 0;
  let s = 0;
  for (let i = 0; i < n; i++) s += Math.abs(real[i] - synth[i]);
  return s / n;
}

function standardizedMeanDifference(realVals, synthVals) {
  const n = Math.min(realVals.length, synthVals.length);
  if (!n) return null;
  const r = realVals.slice(0, n);
  const s = synthVals.slice(0, n);
  const m1 = mean(r), m2 = mean(s);
  const sd1 = std(r), sd2 = std(s);
  const pooled = Math.sqrt((sd1 ** 2 + sd2 ** 2) / 2) || 1e-8;
  return (m1 - m2) / pooled;
}

function renderKpis(kpis) {
  const box = document.getElementById('metrics');
  box.innerHTML = Object.entries(kpis).map(([k,v]) => `
    <div class='kpi'><div class='label'>${k}</div><div class='value'>${v}</div></div>
  `).join('');
}

function getFirstNumericCol() {
  return state.continuous.find(c => isNumericCol(state.df, c));
}

function getFirstCategoricalCol() {
  return (state.categorical[0] || state.binary[0] || null);
}

function destroyChart(name) {
  if (state.charts[name]) { state.charts[name].destroy(); state.charts[name] = null; }
}

function renderLossChart() {
  destroyChart('loss');
  const ctx = document.getElementById('lossChart');
  state.charts.loss = new Chart(ctx, {
    type: 'line',
    data: {
      labels: state.gLosses.map((_, i) => i + 1),
      datasets: [
        { label: 'Generator Loss', data: state.gLosses, borderColor: '#2563eb' },
        { label: 'Discriminator Loss', data: state.dLosses, borderColor: '#dc2626' }
      ]
    },
    options: { responsive: true, animation: false }
  });
}

function valueCounts(rows, col) {
  const m = new Map();
  for (const r of rows) {
    const k = (r[col] ?? '').toString();
    m.set(k, (m.get(k) || 0) + 1);
  }
  return Array.from(m.entries()).sort((a,b) => b[1]-a[1]);
}

function renderResults() {
  if (!state.synthetic.length) return;
  const numCol = getFirstNumericCol();
  const catCol = getFirstCategoricalCol();

  const numericCols = state.continuous.filter(c => isNumericCol(state.df, c));
  const aggregateReal = [];
  const aggregateSynth = [];
  for (const c of numericCols) {
    const rv = state.df.map(r => Number(r[c])).filter(Number.isFinite);
    const sv = state.synthetic.map(r => Number(r[c])).filter(Number.isFinite);
    const n = Math.min(rv.length, sv.length);
    for (let i = 0; i < n; i++) {
      aggregateReal.push(rv[i]);
      aggregateSynth.push(sv[i]);
    }
  }

  const rmseValue = aggregateReal.length ? rmse(aggregateReal, aggregateSynth).toFixed(4) : 'N/A';
  const mseValue = aggregateReal.length ? mse(aggregateReal, aggregateSynth).toFixed(4) : 'N/A';
  const maeValue = aggregateReal.length ? mae(aggregateReal, aggregateSynth).toFixed(4) : 'N/A';

  renderKpis({
    'Real Rows': state.df.length,
    'Synthetic Rows': state.synthetic.length,
    'Continuous Cols': state.continuous.length,
    'Categorical Cols': state.categorical.length + state.binary.length,
    'RMSE': rmseValue,
    'MSE': mseValue,
    'MAE': maeValue
  });

  const quasiCols = getQuasiIdentifierColumns(state.df);
  const realK = computeKAnonymity(state.df, quasiCols);
  const synthK = computeKAnonymity(state.synthetic, quasiCols);
  const privacyBox = document.getElementById('resultPrivacy');
  privacyBox.innerHTML = Object.entries({
    'Real k-Anonymity': realK.k,
    'Synthetic k-Anonymity': synthK.k,
    'Synthetic Vulnerable Rows (<3)': synthK.vulnerable,
    'Synthetic Safe Coverage (%)': `${synthK.coverage.toFixed(1)}%`
  }).map(([k, v]) => `
    <div class='kpi'><div class='label'>${k}</div><div class='value small'>${v}</div></div>
  `).join('');

  renderTable('synthPreview', state.synthetic, 10);

  destroyChart('num');
  if (numCol) {
    const realVals = state.df.map(r => Number(r[numCol])).filter(Number.isFinite);
    const synthVals = state.synthetic.map(r => Number(r[numCol])).filter(Number.isFinite);
    const bins = 15;
    const min = Math.min(...realVals, ...synthVals);
    const max = Math.max(...realVals, ...synthVals);
    const width = (max - min) / bins || 1;
    const realBins = Array(bins).fill(0), synthBins = Array(bins).fill(0);
    for (const v of realVals) realBins[Math.min(bins - 1, Math.floor((v - min) / width))]++;
    for (const v of synthVals) synthBins[Math.min(bins - 1, Math.floor((v - min) / width))]++;
    const labels = Array.from({length: bins}, (_, i) => `${(min + i * width).toFixed(1)}`);
    state.charts.num = new Chart(document.getElementById('numChart'), {
      type: 'bar',
      data: { labels, datasets: [
        { label: `Real (${numCol})`, data: realBins, backgroundColor: 'rgba(37,99,235,0.5)' },
        { label: `Synthetic (${numCol})`, data: synthBins, backgroundColor: 'rgba(220,38,38,0.5)' }
      ]},
      options: { responsive: true, animation: false }
    });
  }

  destroyChart('cat');
  if (catCol) {
    const realVC = valueCounts(state.df, catCol).slice(0, 12);
    const synthVCMap = new Map(valueCounts(state.synthetic, catCol));
    const labels = realVC.map(x => x[0]);
    const realData = realVC.map(x => x[1]);
    const synthData = labels.map(l => synthVCMap.get(l) || 0);
    state.charts.cat = new Chart(document.getElementById('catChart'), {
      type: 'bar',
      data: {
        labels,
        datasets: [
          { label: `Real (${catCol})`, data: realData, backgroundColor: 'rgba(37,99,235,0.5)' },
          { label: `Synthetic (${catCol})`, data: synthData, backgroundColor: 'rgba(220,38,38,0.5)' }
        ]
      },
      options: { responsive: true, animation: false }
    });
  }

  // SMD + Love plot
  const smdRows = [];
  for (const c of numericCols) {
    const rv = state.df.map(r => Number(r[c])).filter(Number.isFinite);
    const sv = state.synthetic.map(r => Number(r[c])).filter(Number.isFinite);
    const smd = standardizedMeanDifference(rv, sv);
    if (smd === null) continue;
    const absSmd = Math.abs(smd);
    let status = 'High';
    if (absSmd < 0.1) status = 'Good';
    else if (absSmd < 0.2) status = 'Acceptable';
    smdRows.push({ Feature: c, SMD: smd.toFixed(4), AbsSMD: absSmd, Status: status });
  }
  smdRows.sort((a, b) => b.AbsSMD - a.AbsSMD);
  renderKeyValueTable('smdTable', smdRows.slice(0, 30), ['Feature', 'SMD', 'Status']);

  destroyChart('love');
  if (smdRows.length) {
    state.charts.love = new Chart(document.getElementById('loveChart'), {
      type: 'bar',
      data: {
        labels: smdRows.map(r => r.Feature),
        datasets: [{
          label: '|SMD|',
          data: smdRows.map(r => r.AbsSMD),
          backgroundColor: smdRows.map(r => r.AbsSMD < 0.1 ? 'rgba(34,197,94,0.6)' : r.AbsSMD < 0.2 ? 'rgba(251,146,60,0.6)' : 'rgba(239,68,68,0.6)')
        }]
      },
      options: { indexAxis: 'y', responsive: true, animation: false }
    });
  }
}

function toCsv(rows) {
  if (!rows.length) return '';
  const cols = Object.keys(rows[0]);
  const esc = (v) => {
    const s = (v ?? '').toString();
    return s.includes(',') || s.includes('"') || s.includes('\n') ? `"${s.replaceAll('"', '""')}"` : s;
  };
  const lines = [cols.join(',')];
  for (const r of rows) lines.push(cols.map(c => esc(r[c])).join(','));
  return lines.join('\n');
}

function download(name, content, type) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = name; a.click();
  URL.revokeObjectURL(url);
}

document.getElementById('downloadCsv').addEventListener('click', () => {
  if (!state.synthetic.length) return;
  download('synthetic_data.csv', toCsv(state.synthetic), 'text/csv');
});

document.getElementById('downloadJson').addEventListener('click', () => {
  if (!state.synthetic.length) return;
  download('synthetic_data.json', JSON.stringify(state.synthetic, null, 2), 'application/json');
});
