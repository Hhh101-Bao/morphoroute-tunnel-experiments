<#
.SYNOPSIS
    MorphoRoute-Tunnel 一键运行脚本（Windows / PowerShell）。

.DESCRIPTION
    统一封装 19 个独立实验入口、全量空运行检查、单元测试和 WSL2 环境安装。
    在 Windows 上非 --dry-run 的运行会按工程设计自动转交给 WSL2 Ubuntu 执行。

.EXAMPLE
    .\run.ps1 list
    .\run.ps1 dry 05
    .\run.ps1 dryall
    .\run.ps1 smoke 05
    .\run.ps1 run 09 -Profile local_6gb -DataRoot "E:\data\铁路隧道多病害实例分割_原始影像严格划分_v1"
    .\run.ps1 test
    .\run.ps1 env
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('list', 'dry', 'dryall', 'smoke', 'run', 'test', 'env')]
    [string]$Action = 'list',

    [Parameter(Position = 1)]
    [string]$Experiment = '05',

    [ValidateSet('local_6gb', 'local_6gb_fallback', 'sop_full', 'autodl_5090_32gb')]
    [string]$Profile = 'local_6gb',

    [string]$DataRoot = $env:MORPHOROUTE_DATA_ROOT,

    [string]$OutputRoot = $env:MORPHOROUTE_OUTPUT_ROOT,

    [string]$Python = $env:MORPHOROUTE_PYTHON
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = $PSScriptRoot
$EntryDir = Join-Path $ProjectRoot '19 个独立实验运行入口'
$TestDir = Join-Path $ProjectRoot '61 项自动化测试'

# 实验编号 -> (英文入口, 中文入口)
$EntryMap = [ordered]@{
    '00'  = @('exp00_preregister.py',   '00预注册与实验冻结.py')
    '01'  = @('exp01_data_audit.py',    '01数据清单、去重与五类完整性审计.py')
    '02'  = @('exp02_yolo_original.py', '02YOLOv8n-seg 原实验条件复现.py')
    '03'  = @('exp03_yolo_strict.py',   '03YOLOv8n-seg 严格划分纠正复现.py')
    '04'  = @('exp04_mask_rcnn.py',     '04Mask R-CNN 两阶段强基线实验.py')
    '05'  = @('exp05_mask2former.py',   '05同骨干 Mask2Former 共享主基线实验.py')
    '06'  = @('exp06_resource_matched.py', '06容量匹配与 FLOPs 匹配共享模型实验.py')
    '07'  = @('exp07_routing_controls.py', '07普通 MoE、类别硬路由与形态路由对照实验.py')
    '08a' = @('exp08a_gradient_conflict.py', '08A梯度冲突诊断 M1.py')
    '08b' = @('exp08b_performance_diagnosis.py', '08B共享、独立、硬拆分与 PCGrad 性能诊断 M2.py')
    '08c' = @('exp08c_morphology_association.py', '08C梯度冲突与连续形态关联 M3.py')
    '08d' = @('exp08d_expert_intervention.py', '08D专家交换、关闭与输入交换 M4.py')
    '09'  = @('exp09_ablation.py',      '09B0–B11 完整消融实验.py')
    '10'  = @('exp10_freeze.py',        '10唯一候选选择与 D4 前冻结.py')
    '11'  = @('exp11_external_test.py', '11D4 一次性外部测试与效能门 E.py')
    '12'  = @('exp12_reliability.py',   '12校准、不确定性与选择性人工复核.py')
    '13'  = @('exp13_quantification.py', '13病害长度、正交宽度与面积量化.py')
    '14'  = @('exp14_sequence_efficiency.py', '14完整序列推理与效率测试.py')
    '15'  = @('exp15_statistics.py',    '15五种子统计、配对 Bootstrap 与论文结论锁定.py')
}

function Normalize-ExperimentId {
    param([string]$Value)

    $clean = $Value.Trim().ToLowerInvariant()
    $clean = $clean -replace '^exp[-_]?', ''
    if ($clean -match '^\d$') {
        return '0' + $clean
    }
    return $clean
}

function Resolve-Entry {
    param([string]$Value)

    $id = Normalize-ExperimentId $Value
    if (-not $EntryMap.Contains($id)) {
        $known = ($EntryMap.Keys | ForEach-Object { 'EXP-' + $_.ToUpperInvariant() }) -join ', '
        throw "无法识别的实验编号 '$Value'。可用编号: $known"
    }
    return @{
        Id      = $id
        Path    = Join-Path $EntryDir $EntryMap[$id][0]
        Chinese = $EntryMap[$id][1]
        English = $EntryMap[$id][0]
    }
}

function Show-Command {
    param([string]$Executable, [string[]]$Arguments)

    $quoted = $Arguments | ForEach-Object {
        if ($_ -match '\s') { '"' + $_ + '"' } else { $_ }
    }
    Write-Host ('> ' + $Executable + ' ' + ($quoted -join ' ')) -ForegroundColor DarkGray
}

if (-not $DataRoot) {
    $DataRoot = Join-Path $ProjectRoot 'datasets\铁路隧道多病害实例分割_原始影像严格划分_v1'
}
if (-not $OutputRoot) {
    $OutputRoot = Join-Path $ProjectRoot 'outputs'
}
if (-not $Python) {
    $Python = 'python'
}

$CommonArguments = @(
    '--profile', $Profile,
    '--dataset-root', $DataRoot,
    '--output-root', $OutputRoot
)

Write-Host 'MorphoRoute-Tunnel 实验工程' -ForegroundColor Cyan
Write-Host ("  工程根目录 : {0}" -f $ProjectRoot)
Write-Host ("  Python     : {0}" -f $Python)
Write-Host ("  档位       : {0}" -f $Profile)
Write-Host ("  数据目录   : {0}" -f $DataRoot)
Write-Host ("  输出目录   : {0}" -f $OutputRoot)

switch ($Action) {
    'list' {
        Write-Host ''
        foreach ($key in $EntryMap.Keys) {
            Write-Host ('EXP-' + $key.ToUpperInvariant() + '  ' + $EntryMap[$key][1] + '   (' + $EntryMap[$key][0] + ')')
        }
        Write-Host ''
        Write-Host '示例: .\run.ps1 dry 05     .\run.ps1 smoke 05     .\run.ps1 run 09'
    }

    'dry' {
        $entry = Resolve-Entry $Experiment
        $arguments = @($entry.Path) + $CommonArguments + @('--dry-run', '--no-auto-run-prerequisites')
        Show-Command $Python $arguments
        & $Python @arguments
        exit $LASTEXITCODE
    }

    'dryall' {
        $failures = @()
        $previousPreference = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try {
            foreach ($key in $EntryMap.Keys) {
                $entryPath = Join-Path $EntryDir $EntryMap[$key][0]
                $arguments = @($entryPath) + $CommonArguments + @('--dry-run', '--no-auto-run-prerequisites')
                $output = & $Python @arguments 2>&1 | Out-String
                if ($LASTEXITCODE -eq 0) {
                    Write-Host ('PASS  EXP-{0}' -f $key.ToUpperInvariant()) -ForegroundColor Green
                }
                else {
                    $match = ($output -split '\r?\n' | Where-Object { $_ -match 'ERROR' } | Select-Object -First 1)
                    if ($match) { $match = $match -replace '^.*?ERROR', 'ERROR' }
                    Write-Host ('FAIL  EXP-{0}  {1}' -f $key.ToUpperInvariant(), $match.Trim()) -ForegroundColor Red
                    $failures += $key
                }
            }
        }
        finally {
            $ErrorActionPreference = $previousPreference
        }
        if ($failures.Count -gt 0) {
            Write-Host ''
            Write-Host ('空运行失败 {0} 个实验: {1}' -f $failures.Count, ($failures -join ', ')) -ForegroundColor Red
            exit 1
        }
        Write-Host ''
        Write-Host '全部 19 个实验入口空运行通过。' -ForegroundColor Green
    }

    'smoke' {
        $entry = Resolve-Entry $Experiment
        $arguments = @($entry.Path) + $CommonArguments + @('--smoke-test')
        Show-Command $Python $arguments
        Write-Host '提示: smoke 只验证代码链，结果写入 outputs\smoke_runs，不能进入论文统计。' -ForegroundColor Yellow
        & $Python @arguments
        exit $LASTEXITCODE
    }

    'run' {
        $entry = Resolve-Entry $Experiment
        if (-not (Test-Path -LiteralPath $DataRoot)) {
            throw "数据目录不存在: $DataRoot。请用 -DataRoot 指定严格划分数据集目录。"
        }
        $arguments = @($entry.Path) + $CommonArguments
        Show-Command $Python $arguments
        Write-Host '提示: Windows 上正式运行会通过 wsl.exe 转入 WSL2 Ubuntu 的隔离环境。' -ForegroundColor Yellow
        & $Python @arguments
        exit $LASTEXITCODE
    }

    'test' {
        $arguments = @('-m', 'unittest', 'discover', '-s', $TestDir, '-t', $TestDir, '-v')
        Show-Command $Python $arguments
        & $Python @arguments
        exit $LASTEXITCODE
    }

    'env' {
        $installer = Join-Path $ProjectRoot '工程辅助运行脚本\实验环境自动检测与安装.py'
        $arguments = @($installer, '--kind', 'all')
        Show-Command $Python $arguments
        & $Python @arguments
        exit $LASTEXITCODE
    }
}
