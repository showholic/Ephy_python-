Get-ChildItem -Filter '*.AVI' | %{
ffprobe.exe -i $_.Name -show_frames |sls media_type=video -context 0,4 |
%{
$_.context.PostContext[3] -replace {.*=}}|
Out-File $_.Name.replace('.AVI','_ts.txt')}