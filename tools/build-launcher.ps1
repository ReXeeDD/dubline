$ErrorActionPreference = 'Stop'
$Project = Split-Path -Parent $PSScriptRoot
$Compiler = Join-Path $env:WINDIR 'Microsoft.NET\Framework\v4.0.30319\csc.exe'
$Source = Join-Path $PSScriptRoot 'DublineLauncher.cs'
$Output = Join-Path $Project 'Dubline.exe'

& $Compiler /nologo /target:exe /platform:anycpu /optimize+ "/out:$Output" $Source
if ($LASTEXITCODE -ne 0) {
    throw "Compiling Dubline.exe failed with exit code $LASTEXITCODE."
}
