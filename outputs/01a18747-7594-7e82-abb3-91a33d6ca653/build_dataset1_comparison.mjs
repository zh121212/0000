import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

// 只读取既有基线记录，所有新文件均写入独立输出目录。
const root = 'E:/小论文/HPA/TimeXer_TaskA_Cloud';
const outDir = path.dirname(fileURLToPath(import.meta.url));
const sourceDir = path.join(root, '03_实验结果/01_对比实验');
const readJson = async p => JSON.parse(await fs.readFile(p, 'utf8'));
const digest = async p => crypto.createHash('sha256').update(await fs.readFile(p)).digest('hex');
const cfgPath = path.join(root, '04_实验配置/protocol.json');
const selectionPath = path.join(root, '03_实验结果/运行记录/selection.json');
const metricsPath = path.join(sourceDir, 'all_seed_metrics.csv');
const packingPath = path.join(root, '01_数据集/03_划分打包/packing_audit.json');
const cfg = await readJson(cfgPath);
const selection = await readJson(selectionPath);
const packing = await readJson(packingPath);
const originalHashes = {};
for (const p of [cfgPath, selectionPath, metricsPath, packingPath]) originalHashes[p] = await digest(p);
const imported = await Workbook.fromCSV(await fs.readFile(metricsPath, 'utf8'), { sheetName: 'Imported' });
const raw = imported.worksheets.getItemAt(0).getUsedRange().values;
const header = raw[0];
const sources = raw.slice(1).filter(row => row.some(v => v !== null && v !== ''));
const sourceRecords = sources.map(row => Object.fromEntries(header.map((k, i) => [k, row[i]])));
assert.equal(sourceRecords.length, 36);
const toNum = v => v === null || v === '' || v === undefined ? null : Number(v);
const close = (a, b, tolerance = 2e-12) => assert.ok(Math.abs(a-b) < tolerance, `${a} != ${b}`);
const mean = xs => xs.reduce((a,b) => a+b,0)/xs.length;
const sd = xs => Math.sqrt(xs.reduce((s,v)=>s+(v-mean(xs))**2,0)/(xs.length-1));
const metrics = ['MAE', 'MAPE', 'RMSE', 'skill_vs_persistence'];
const runRecords = [];
let maxRecalculationDifference = 0;

// 用35份训练记录及逐时刻预测独立核对数据，绝不重新训练或挑选种子。
for (const model of cfg.models) {
  const selected = selection.models[`A/${model}`];
  for (const seed of cfg.seeds) {
    const dir = path.join(sourceDir, `runs/A_${model}_lr${selected.lr}_s${seed}`);
    const result = await readJson(path.join(dir, 'result.json'));
    const job = await readJson(path.join(dir, 'job.json'));
    const test = await readJson(path.join(dir, 'test_metrics.json'));
    for (const record of [result, job, test]) {
      assert.equal(record.fingerprint, selection.fingerprint);
      assert.equal(record.seed, seed);
      assert.equal(record.model, model);
      assert.equal(record.lr, selected.lr);
    }
    assert.equal(result.status, 'completed');
    assert.equal(test.n, packing.tasks.A.splits.test.samples);
    const epochLines = (await fs.readFile(path.join(dir,'epochs.csv'),'utf8')).trim().split(/\r?\n/);
    const epochHeaders = epochLines[0].split(',');
    const epochs = epochLines.slice(1).map(line => {
      const cells=line.split(',');
      assert.equal(cells.length, epochHeaders.length);
      return Object.fromEntries(epochHeaders.map((key,i)=>[key,Number(cells[i])]));
    });
    const best = epochs.find(e => e.epoch === result.best_epoch);
    close(best.val_RMSE, result.best_val_RMSE);
    close(Math.min(...epochs.map(e=>e.val_RMSE)), result.best_val_RMSE);
    assert.equal(epochs.length,result.epochs_run);
    const source = sourceRecords.find(r=>r.model===model && Number(r.seed)===seed);
    assert.ok(source);
    for (const metric of metrics) close(Number(source[metric]),test[metric]);
    const predictionLines = (await fs.readFile(path.join(dir,'test_predictions.csv'),'utf8')).trim().replace(/^\uFEFF/,'').split(/\r?\n/);
    const names = predictionLines[0].split(',');
    const oi=names.indexOf('observed'), pi=names.indexOf('predicted'), bi=names.indexOf('persistence');
    assert.ok(oi>=0 && pi>=0 && bi>=0);
    let abs=0, sq=0, ape=0, baseSq=0;
    for (const line of predictionLines.slice(1)) {
      const cells=line.split(',');
      const observed=Number(cells[oi]), error=Number(cells[pi])-observed;
      abs+=Math.abs(error); sq+=error**2; ape+=Math.abs(error/observed);
      baseSq+=(Number(cells[bi])-observed)**2;
    }
    const n=predictionLines.length-1;
    assert.equal(n,test.n);
    const recalculated={MAE:abs/n,RMSE:Math.sqrt(sq/n),MAPE:ape/n*100,skill_vs_persistence:1-sq/baseSq};
    for (const metric of metrics) {
      maxRecalculationDifference=Math.max(maxRecalculationDifference,Math.abs(recalculated[metric]-test[metric]));
      close(recalculated[metric],test[metric],2e-9);
    }
    runRecords.push({model,seed,lr:selected.lr,valMAE:best.val_MAE,valRMSE:result.best_val_RMSE,...Object.fromEntries(metrics.map(m=>[m,test[m]])),bestEpoch:result.best_epoch,epochsRun:result.epochs_run,parameters:result.parameters,trainingSeconds:result.training_seconds});
  }
}
assert.equal(runRecords.length,35);
const sortedModels = [...cfg.models].sort((a,b)=>mean(runRecords.filter(r=>r.model===a).map(r=>r.RMSE))-mean(runRecords.filter(r=>r.model===b).map(r=>r.RMSE)));
runRecords.sort((a,b)=>sortedModels.indexOf(a.model)-sortedModels.indexOf(b.model)||cfg.seeds.indexOf(a.seed)-cfg.seeds.indexOf(b.seed));
const reference=sourceRecords.find(r=>r.model==='Persistence');
assert.ok(reference);

const wb=Workbook.create();
const summary=wb.worksheets.add('对比汇总');
const detail=wb.worksheets.add('五种子明细');
const font={name:'Arial',size:11,color:'#172B3A'};
function base(sheet, range) {
  sheet.showGridLines=false;
  sheet.getRange(range).format.font=font;
  sheet.getRange(range).format.rowHeight=23;
  sheet.getRange(range).format.verticalAlignment='center';
}
function title(sheet, cell, text) {
  sheet.getRange(cell).values=[[text]];
  sheet.getRange(cell).format.font={...font,size:15,bold:true};
  sheet.getRange(cell).format.rowHeight=32;
}
function heading(sheet, range) {
  sheet.getRange(range).format.fill='#26495D';
  sheet.getRange(range).format.font={...font,bold:true,color:'#FFFFFF'};
  sheet.getRange(range).format.horizontalAlignment='center';
  sheet.getRange(range).format.rowHeight=30;
}
base(summary,'A1:I57'); base(detail,'A1:N48');
summary.tabColor='#26495D';
title(summary,'A2','数据集1基线模型对比');
summary.getRange('A3').values=[['任务A：淇河（镇平）与十二里河历史，预测十二里河30分钟后的闸前水位。']];
summary.getRange('A4').values=[['水位误差单位为m。均值与样本标准差分别列示。模型按测试RMSE均值排序。']];
summary.getRange('A5').values=[['七模型各5个种子，共35组最终结果。粗体表示七模型中该项最优均值。']];
summary.getRange('A3:A5').format.font={...font,size:11,color:'#536773'};
const summaryHeaders=['模型','统计','验证RMSE (m)','测试MAE (m)','测试MAPE (%)','测试RMSE (m)','测试技能分数','学习率','种子数'];
summary.getRange('A7:I7').values=[summaryHeaders];heading(summary,'A7:I7');
detail.getRange('A1').values=[['来源：原基线all_seed_metrics.csv及各运行目录中的result.json、epochs.csv和test_metrics.json。']];
detail.getRange('A2').values=[['数值保留源记录精度。验证指标取各次训练验证RMSE最低轮次，测试指标来自该轮权重。']];
detail.getRange('A3').values=[['源工程：E:/小论文/HPA/TimeXer_TaskA_Cloud；原结果统一指纹：'+selection.fingerprint]];
const detailHeaders=['模型','种子','学习率','验证MAE (m)','验证RMSE (m)','测试MAE (m)','测试MAPE (%)','测试RMSE (m)','测试技能分数','最优轮次','实际训练轮数','参数量','训练耗时 (s)','训练状态'];
detail.getRange('A5:N5').values=[detailHeaders];heading(detail,'A5:N5');
const detailRows=runRecords.map(r=>[r.model,r.seed,r.lr,r.valMAE,r.valRMSE,r.MAE,r.MAPE,r.RMSE,r.skill_vs_persistence,r.bestEpoch,r.epochsRun,r.parameters,r.trainingSeconds,'已完成']);
detail.getRange('A6:N40').values=detailRows;
detail.getRange('C6:C40').setNumberFormat('0.0000');
detail.getRange('D6:I40').setNumberFormat('0.000000000');
detail.getRange('M6:M40').setNumberFormat('0.00');
detail.getRange('B6:M40').format.horizontalAlignment='right';
detail.getRange('A42:I42').values=[['保持不变预测',null,null,null,null,Number(reference.MAE),Number(reference.MAPE),Number(reference.RMSE),Number(reference.skill_vs_persistence)]];
detail.getRange('D42:I42').setNumberFormat('0.000000000');
detail.getRange('A44').values=[['确定性参考不运行5个随机种子，因此其标准差不填0。']];
detail.getRange('A46').values=[['训练耗时受云端并行资源竞争影响，不用于严格的模型效率排名。']];
detail.freezePanes.freezeRows(5);detail.freezePanes.freezeColumns(2);
detail.tables.add('A5:N40',true,'SeedDetails');
const expectedSummary=[];
const meanRows=[];
for (let i=0;i<sortedModels.length;i++) {
  const model=sortedModels[i], first=6+i*5,last=first+4,row=8+i*2;
  const records=runRecords.filter(r=>r.model===model);
  meanRows.push(row);
  summary.getRange(`A${row}:I${row+1}`).values=[[model,'均值',null,null,null,null,null,records[0].lr,5],['','标准差',null,null,null,null,null,null,null]];
  const cols=['E','F','G','H','I'];
  for (let j=0;j<cols.length;j++) {
    const target=String.fromCharCode(67+j);
    summary.getRange(`${target}${row}`).formulas=[[`=AVERAGE('五种子明细'!${cols[j]}${first}:${cols[j]}${last})`]];
    summary.getRange(`${target}${row+1}`).formulas=[[`=STDEV.S('五种子明细'!${cols[j]}${first}:${cols[j]}${last})`]];
  }
  summary.getRange(`A${row}:I${row}`).format.fill=i%2===0?'#EDF3F6':'#FFFFFF';
  summary.getRange(`A${row+1}:I${row+1}`).format.font={...font,color:'#697984'};
  summary.getRange(`A${row+1}:I${row+1}`).format.borders={bottom:{style:'thin',color:'#D9E2E7'}};
  expectedSummary.push({model,row,mean:[mean(records.map(r=>r.valRMSE)),...metrics.map(m=>mean(records.map(r=>r[m])))],sd:[sd(records.map(r=>r.valRMSE)),...metrics.map(m=>sd(records.map(r=>r[m])))]});
}
const valRefRMSE=0.0070797675264008;
summary.getRange('A23:I23').values=[['保持不变预测','参考',valRefRMSE,Number(reference.MAE),Number(reference.MAPE),Number(reference.RMSE),0,null,null]];
summary.getRange('A23:I23').format.fill='#E7ECEF';
summary.getRange('C8:G23').setNumberFormat('0.000000000');
summary.getRange('H8:H23').setNumberFormat('0.0000');
summary.getRange('C8:I23').format.horizontalAlignment='right';
for(let j=0;j<5;j++) {
  const aggregate=j===4?'MAX':'MIN';
  const col=String.fromCharCode(67+j);
  const bestFormula=`${aggregate}(${meanRows.map(row=>`$${col}$${row}`).join(',')})`;
  for(const row of meanRows) {
    summary.getRange(`${col}${row}`).conditionalFormats.addCustom(`=${col}${row}=${bestFormula}`,{font:{bold:true}});
  }
}
title(summary,'A26','共同实验设置');
const settingRows=[
 ['数据组','数据集1，淇河（镇平）与十二里河，任务A'],
 ['预测对象','十二里河30分钟后的闸前水位，单目标单步预测'],
 ['输入与输出','12×17历史输入，输出1个水位值；不使用未来实测值'],
 ['时间划分','按时间顺序70%训练、10%验证、20%测试'],
 ['标准化','只用训练段拟合物理量均值和标准差，日历保持原编码'],
 ['样本数','训练11,244，验证1,608，测试3,217'],
 ['训练标签时段','2025-01-01 06:00至2025-08-23 11:30'],
 ['验证标签时段','2025-08-23 12:00至2025-09-25 23:30'],
 ['测试标签时段','2025-09-26 00:00至2025-12-02 00:00'],
 ['随机种子',cfg.seeds.join('、')],
 ['优化器','Adam，weight_decay=0.0001'],
 ['训练参数','batch size=64，dropout=0.1，梯度裁剪=5'],
 ['停止规则','最多100轮，验证RMSE连续10轮未改善则早停'],
 ['学习率搜索','种子42搜索0.0001、0.0003、0.001，验证RMSE选优后补齐其余4种子'],
 ['参数选择','每个种子使用自身验证RMSE最低轮次的权重，不按测试成绩选择'],
 ['模型损失','六模型为MSE；TimeFilter为MSE加0.05倍模型辅助损失'],
 ['HPA版本','历史输入适配版，保留模型结构思想，不称原论文实验原样复现'],
 ['技能分数','1－模型MSE／保持不变预测MSE；越大越好，正值优于参考'],
 ['比较范围','原七模型基线。后续weight_decay=0的TimeXer改进结果未并入'],
 ['训练集误差','训练日志保存标准化目标损失，不能与本表原始水位误差混称'],
 ['数据组命名','当前源目录名为“数据集2”，本表“数据集1”沿用对话中的淇河组编号'],
 ['汇总精度','由35组原始精度数值计算，显示9位小数，未采用已舍入的汇总CSV']
];
for(let i=0;i<settingRows.length;i++){
  const row=28+i;
  summary.getRange(`A${row}:B${row}`).values=[settingRows[i]];
  summary.getRange(`A${row}`).format.font={...font,bold:true};
}
summary.getRange('A52').values=[['验证集最低均值为iTransformer，测试集最低均值为TimeFilter。']];
summary.getRange('A53').values=[['只有TimeFilter的测试技能分数为正。不同时间段指标不能直接相减解释为改进。']];
// 使用足够宽的列保留单位、符号和9位小数。
summary.getRange('A1:A57').format.columnWidth=24;
summary.getRange('B1:B57').format.columnWidth=12;
summary.getRange('C1:G57').format.columnWidth=22;
summary.getRange('H1:H57').format.columnWidth=14;
summary.getRange('I1:I57').format.columnWidth=11;
summary.getRange('A28:A49').format.rowHeight=25;
detail.getRange('A1:A48').format.columnWidth=22;
detail.getRange('B1:C48').format.columnWidth=12;
detail.getRange('D1:I48').format.columnWidth=21;
detail.getRange('J1:L48').format.columnWidth=16;
detail.getRange('M1:N48').format.columnWidth=19;

wb.recalculate();
for(const item of expectedSummary){
  const actualMean=summary.getRange(`C${item.row}:G${item.row}`).values[0];
  const actualSd=summary.getRange(`C${item.row+1}:G${item.row+1}`).values[0];
  for(let j=0;j<5;j++){close(Number(actualMean[j]),item.mean[j]);close(Number(actualSd[j]),item.sd[j]);}
}
// 验证编辑明细能实际触发均值重算，随后恢复原始记录。
const originalFirst=detail.getRange('H6').values[0][0];
const before=summary.getRange('F8').values[0][0];
detail.getRange('H6').values=[[originalFirst+0.001]];
wb.recalculate();close(Number(summary.getRange('F8').values[0][0]),Number(before)+0.001/5);
detail.getRange('H6').values=[[originalFirst]];
wb.recalculate();close(Number(summary.getRange('F8').values[0][0]),Number(before));
const inspection=await wb.inspect({kind:'table',range:'对比汇总!A7:I11',include:'values,formulas',tableMaxRows:5,tableMaxCols:9,maxChars:3500});
await fs.writeFile(path.join(outDir,'summary_inspection.ndjson'),inspection.ndjson);
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:30},summary:'formula errors'});
await fs.writeFile(path.join(outDir,'formula_scan.ndjson'),errors.ndjson);
console.log('FORMULA_SCAN',errors.ndjson);
for(const [name,range,file] of [['对比汇总','A1:I24','summary_preview.png'],['对比汇总','A26:I54','settings_preview.png'],['五种子明细','A1:N12','detail_preview.png']]){
  const blob=await wb.render({sheetName:name,range,scale:1.5,format:'png'});
  await fs.writeFile(path.join(outDir,file),new Uint8Array(await blob.arrayBuffer()));
}
const outputPath=path.join(outDir,'数据集1_七模型同设置对比结果.xlsx');
await (await SpreadsheetFile.exportXlsx(wb)).save(outputPath);
for(const [p,hash] of Object.entries(originalHashes)) assert.equal(await digest(p),hash);
const report={date:'2026-10-29',models:sortedModels,seeds:cfg.seeds,finalRuns:35,consistentFingerprint:selection.fingerprint,testSamples:3217,maxMetricRecalculationDifference:maxRecalculationDifference,summaryMeanAndSdVerified:true,liveFormulaRecalculationVerified:true,originalFilesUnchanged:true,sourceHashes:originalHashes,outputPath};
await fs.writeFile(path.join(outDir,'verification.json'),JSON.stringify(report,null,2));
console.log(JSON.stringify(report,null,2));
