param(
    [string]$Root = "C:\Users\frisa\Desktop\BenchmarkingAI",
    [string]$StartAt = "00:00:00",
    [string]$RtspBase = "rtsp://127.0.0.1:8554",
    [switch]$Transcode
)

$ErrorActionPreference = "Stop"

$videos = @{
    "cam_02" = "recordings\recordings\Camera_2_2.3_20260506_131002.mp4"
    "cam_03" = "recordings\recordings\Camera_3_2.4_20260506_131002.mp4"
    "cam_05" = "recordings\recordings\Camera_5_2.6_20260506_131002.mp4"
    "cam_07" = "recordings\recordings\Camera_7_2.11_20260506_131002.mp4"
}

foreach ($item in $videos.GetEnumerator()) {
    $cam = $item.Key
    $video = Join-Path $Root $item.Value
    if (-not (Test-Path $video)) {
        throw "Video introuvable pour $cam : $video"
    }

    if ($Transcode) {
        $ffArgs = @(
            "-stream_loop", "-1",
            "-re",
            "-ss", $StartAt,
            "-i", "`"$video`"",
            "-an",
            "-vf", "`"scale=704:576,fps=25,format=yuv420p`"",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-tune", "zerolatency",
            "-g", "25",
            "-keyint_min", "25",
            "-bf", "0",
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            "$RtspBase/$cam"
        )
    } else {
        $ffArgs = @(
            "-stream_loop", "-1",
            "-re",
            "-ss", $StartAt,
            "-i", "`"$video`"",
            "-an",
            "-c:v", "copy",
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            "$RtspBase/$cam"
        )
    }

    $argLine = $ffArgs -join " "
    Write-Host "[INFO] Publishing $cam from $StartAt -> $RtspBase/$cam"
    Start-Process -FilePath "powershell.exe" -ArgumentList "-NoExit", "-Command", "cd `"$Root`"; ffmpeg $argLine"
}

Write-Host "[INFO] 4 replay publishers started. Check HLS:"
Write-Host "       http://127.0.0.1:8888/cam_02/"
Write-Host "       http://127.0.0.1:8888/cam_03/"
Write-Host "       http://127.0.0.1:8888/cam_05/"
Write-Host "       http://127.0.0.1:8888/cam_07/"
