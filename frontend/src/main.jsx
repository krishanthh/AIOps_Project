import React, {useEffect, useRef, useState} from 'react';
import {createRoot} from 'react-dom/client';
import './style.css';

const API = import.meta.env.VITE_API_URL || '/api';
const names = {glioma:'Glioma',meningioma:'Meningioma',notumor:'No tumour',pituitary:'Pituitary tumour'};
async function request(path, options={}) {
  const response = await fetch(API + path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : `Request failed (${response.status})`);
  return data;
}
function Empty({children}) {return <div className="empty">{children}</div>}
function App() {
  const [page,setPage]=useState('Analyse MRI'), [file,setFile]=useState(null), [preview,setPreview]=useState('');
  const [result,setResult]=useState(null), [error,setError]=useState(''), [busy,setBusy]=useState(false);
  const [ready,setReady]=useState(false), [pipeline,setPipeline]=useState(null), [models,setModels]=useState(null);
  const [drift,setDrift]=useState(null), [log,setLog]=useState(''), [explain,setExplain]=useState(true);
  const [links,setLinks]=useState({mlflow:'http://localhost:5000',grafana:'http://localhost:3101',prometheus:'http://localhost:9090'});
  const [refreshError,setRefreshError]=useState(''); const input=useRef();
  async function refresh() {
    const results=await Promise.allSettled([request('/ready'),request('/api/v1/pipeline'),request('/api/v1/models'),request('/api/v1/drift'),request('/api/v1/links')]);
    setReady(results[0].status==='fulfilled');
    if(results[1].status==='fulfilled')setPipeline(results[1].value);
    if(results[2].status==='fulfilled')setModels(results[2].value);
    if(results[3].status==='fulfilled')setDrift(results[3].value);
    if(results[4].status==='fulfilled')setLinks(results[4].value);
    setRefreshError(results.slice(1).some(r=>r.status==='rejected')?'Some operational data could not be refreshed. Check the backend connection.':'');
  }
  useEffect(()=>{refresh();const timer=setInterval(refresh,5000);return()=>clearInterval(timer)},[]);
  useEffect(()=>{if(!file){setPreview('');return}const url=URL.createObjectURL(file);setPreview(url);return()=>URL.revokeObjectURL(url)},[file]);
  function choose(candidate) {
    setError('');setResult(null);
    if(!candidate)return;
    if(!['image/jpeg','image/png'].includes(candidate.type)){setError('Please choose a JPEG or PNG image.');return}
    if(candidate.size>10*1024*1024){setError('This image exceeds the 10 MB limit.');return}
    setFile(candidate);
  }
  async function analyse() {
    setBusy(true);setError('');setResult(null);
    const body=new FormData();body.append('image',file);
    try{setResult(await request(`/api/v1/predict?explain=${explain}`,{method:'POST',body}));refresh()}
    catch(e){setError(e.message)}finally{setBusy(false)}
  }
  async function runPipeline(){setError('');try{await request('/api/v1/pipeline/run',{method:'POST'});refresh()}catch(e){setError(e.message)}}
  return <div className="shell">
    <aside><a className="brand" href="#" onClick={e=>{e.preventDefault();setPage('Analyse MRI')}}><span className="brand-icon">B</span><span>BrainScope<small>MRI INTELLIGENCE</small></span></a>
      <div className="nav-label">WORKSPACE</div><nav>{['Analyse MRI','ML pipeline','Model evaluation','Observability'].map((p,i)=><button className={page===p?'selected':''} key={p} onClick={()=>{setPage(p);setError('')}}><span className="nav-icon">{['◉','⌘','▥','⌁'][i]}</span>{p}</button>)}</nav>
      <div className="aside-bottom"><span className="local-dot"/> Local deployment<p>DA3408 · AIOps Project</p><small>Krishanthh Mohan · DA24B044</small></div>
    </aside>
    <main><header><span>Workspace <span className="slash">/</span> {page}</span><span className={'status '+(ready?'up':'down')}><i/>{ready?'Model connected':'Model unavailable'}</span></header>
    <div className="content"><div className="eyebrow">BRAIN MRI CLASSIFICATION</div><h1>{page==='Analyse MRI'?'A clearer view of your MRI.':page}</h1><p className="subtitle">{page==='Analyse MRI'?'Upload an MRI image to explore a model prediction and its visual explanation.':page==='ML pipeline'?'Follow each stage, from raw images to a packaged model.':page==='Model evaluation'?'Compare real experiments and inspect held-out performance.':'Inspect application health, pipeline throughput, and distribution changes.'}</p>
    {error&&<div role="alert" className="alert">{error}</div>}{refreshError&&<div role="status" className="alert">{refreshError}</div>}
    {page==='Analyse MRI'&&<><div className="analysis-grid"><section className="card"><div className="card-heading"><span className="step">01</span><h2>Input image</h2><span className="tag">JPEG / PNG</span></div>
      <input ref={input} type="file" accept="image/jpeg,image/png" aria-label="Choose brain MRI image" onChange={e=>choose(e.target.files[0])} disabled={busy} hidden/>
      <button className={'dropzone '+(preview?'has-preview':'')} disabled={busy} onClick={()=>input.current.click()} onDragOver={e=>e.preventDefault()} onDrop={e=>{e.preventDefault();if(!busy)choose(e.dataTransfer.files[0])}}>
      {preview?<img src={preview} alt="Selected MRI preview"/>:<><span className="upload-icon">↑</span><strong>Drop an MRI image here</strong><span>or click to browse your files</span><small>JPEG or PNG · up to 10 MB</small></>}</button>
      {file&&<div className="file-info"><span>{file.name}</span><small>{(file.size/1024).toFixed(0)} KB</small></div>}
      <label className="checkbox"><input type="checkbox" checked={explain} onChange={e=>setExplain(e.target.checked)} disabled={busy}/> Include Grad-CAM explanation</label>
      <button className="primary" disabled={!file||busy||!ready} onClick={analyse}>{busy?'Analysing image…':'Analyse MRI'} <span>↗</span></button>
      {!ready&&<small className="hint">Start the model-serving service to enable analysis.</small>}
      </section><section className="card result"><div className="card-heading"><span className="step">02</span><h2>Prediction & explanation</h2></div>
      {busy?<Empty><div className="spinner"/><strong>Reading the image</strong><p>Generating probabilities{explain?' and a visual explanation':''}…</p></Empty>:result?<>
        <div className="prediction"><small>MODEL PREDICTION</small><h2>{names[result.predicted_class]}</h2><span>{(result.confidence*100).toFixed(1)}% model score</span>{result.experimental&&<p className="experimental">Experimental model · validation target not met</p>}</div>
        <div className="probabilities">{Object.entries(result.probabilities).map(([c,v])=><div key={c}><div><span>{names[c]}</span><strong>{(v*100).toFixed(1)}%</strong></div><div className="bar"><i style={{width:`${v*100}%`}}/></div></div>)}</div>
        {result.gradcam_base64&&<figure><img src={'data:image/png;base64,'+result.gradcam_base64} alt="Grad-CAM overlay showing regions influencing the predicted class"/><figcaption>Grad-CAM · regions influencing this prediction</figcaption></figure>}
        <div className="result-meta">{(result.inference_time_ms/1000).toFixed(2)} s · Model {result.model_version.slice(0,10)}</div></>:<Empty><div className="empty-scan">◎</div><strong>Your analysis will appear here</strong><p>Select an image and run the model to see<br/>class probabilities and a visual explanation.</p></Empty>}
      </section></div><div className="notice"><span>ⓘ</span><p><strong>An educational research application.</strong> Scores are not calibrated clinical confidence. Grad-CAM shows model attention, not tumour boundaries. Uploaded images are processed in memory and are not saved by the prediction API.</p></div>
      <div className="features"><div><span>4</span><p>classification categories</p></div><div><span>Grad-CAM</span><p>visual model explanations</p></div><div><span>Local</span><p>inference & monitoring</p></div></div></>}
    {page==='ML pipeline'&&<><section className="card"><div className="card-heading"><h2>Pipeline execution</h2><span className="tag">{pipeline?.status||'Loading'}</span><button className="secondary" onClick={runPipeline} disabled={!pipeline?.controls_enabled||pipeline?.status==='running'}>Run pipeline</button></div><p className="hint">{pipeline?.controls_enabled?'One run at a time. Logs update as stages complete.':'Use the local pipeline command or the training container to start a run.'}</p><div className="dag">{pipeline?.dag?.map((s,i)=><React.Fragment key={s}><div className={'stage '+(pipeline.stages[s]?.status||'pending')}><small>0{i+1}</small><strong>{s}</strong><span>{pipeline.stages[s]?.status||'pending'}</span>{pipeline.stages[s]?.seconds!=null&&<small>{pipeline.stages[s].seconds.toFixed(1)} s</small>}{pipeline.stages[s]?.log&&<button onClick={async()=>{try{const r=await fetch(API+'/api/v1/pipeline/log/'+pipeline.stages[s].log);if(!r.ok)throw new Error('Log unavailable');setLog(await r.text())}catch(e){setError(e.message)}}}>View log</button>}</div>{i<pipeline.dag.length-1&&<span className="arrow">→</span>}</React.Fragment>)}</div>{pipeline?.error&&<div className="alert">{pipeline.error}</div>}</section>
    <div className="stats-grid"><section className="card"><small>PREPARATION ENGINE</small><h2>{pipeline?.preparation?.engine||'Not run'}</h2><p>{pipeline?.preparation?.processed??'—'} images processed</p></section><section className="card"><small>THROUGHPUT</small><h2>{pipeline?.preparation?.images_per_second?.toFixed(1)||'—'} <small>images/s</small></h2><p>Measured from the actual Ray run</p></section><section className="card"><small>DATA QUALITY</small><h2>{pipeline?.quality?.exact_duplicates_removed?.length??'—'}</h2><p>Exact duplicate images removed</p></section></div>{log&&<section className="card"><h2>Stage log</h2><pre>{log}</pre></section>}</>}
    {page==='Model evaluation'&&<><section className="card"><div className="card-heading"><h2>Experiment comparison</h2><a href={links.mlflow} target="_blank" rel="noreferrer">Open MLflow ↗</a></div>{models?.experiments?.length?<div className="table-wrap"><table><thead><tr><th>Run</th><th>Learning rate</th><th>Validation macro-F1</th><th>Training time</th></tr></thead><tbody>{models.experiments.map(r=><tr key={r.run_id}><td>{r.run_id.slice(0,12)}</td><td>{r.learning_rate}</td><td>{r.validation_macro_f1.toFixed(4)}</td><td>{r.seconds.toFixed(1)} s</td></tr>)}</tbody></table></div>:<Empty>No experiments recorded yet. Run the training pipeline first.</Empty>}</section>
      <section className="card"><h2>Held-out test performance</h2>{models?.evaluation?<><div className="features"><div><span>{(models.evaluation.accuracy*100).toFixed(1)}%</span><p>accuracy</p></div><div><span>{models.evaluation['macro avg']['f1-score'].toFixed(3)}</span><p>macro-F1</p></div><div><span>{models.evaluation.samples}</span><p>test images</p></div></div><table><thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th></tr></thead><tbody>{Object.keys(names).map(c=><tr key={c}><td>{names[c]}</td>{['precision','recall','f1-score'].map(k=><td key={k}>{models.evaluation[c][k].toFixed(3)}</td>)}</tr>)}</tbody></table><p className="hint">Image-level evaluation. Without patient identifiers, patient independence cannot be established.</p></>:<Empty>No held-out evaluation is available yet.</Empty>}</section></>}
    {page==='Observability'&&<><div className="stats-grid">{[['Grafana','Dashboards & trends',links.grafana],['Prometheus','Metrics & alert rules',links.prometheus],['MLflow','Experiments & artifacts',links.mlflow]].map(([title,desc,url])=><a className="card tool" href={url} target="_blank" rel="noreferrer" key={title}><span>↗</span><h2>{title}</h2><p>{desc}</p></a>)}</div><section className="card"><h2>Input distribution monitor</h2><p>Status: <strong>{drift?.status?.replaceAll('_',' ')||'Loading'}</strong></p><p>{drift?.samples??0} recent images · minimum 20 required</p>{drift?.zscore!=null&&<p>Brightness departure: {drift.zscore.toFixed(2)} baseline standard deviations</p>}<p className="hint">This monitor compares image brightness with the training baseline. A distribution warning does not measure model accuracy. The recent window resets when the API restarts.</p></section><section className="card"><h2>What is monitored</h2><p>Request rates and errors · prediction latency · model availability · per-class prediction counts · pipeline stage duration · Ray preparation throughput · process memory and CPU.</p></section></>}
    <footer>BrainScope <span>Brain Tumour Classification · DA3408</span></footer></div></main></div>
}
createRoot(document.getElementById('root')).render(<App/>);
