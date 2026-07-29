# Phase 4 experiment matrix

Run one case at a time and keep the generated report directory.

## video_linux_ethernet_clean_rtsp_tcp

- family: `video_network`
- network: `clean`
- video: `rtsp_tcp`
- alert: `none`

Network setup:

```bash
sudo tc qdisc del dev eth0 root
```

Run:

```powershell
python3 Phase_3_Fusion_MultiCam/run_live_campaign.py --versions V4 --models yolov8s --formats fp32_engine --cameras cam_02,cam_03,cam_05,cam_07 --duration-min 10 --device cuda:0 --record-fps 25 --object-min-camera-votes 2 --capture-backend gstreamer --gst-protocol tcp --gst-latency-ms 50 --gst-pipeline decodebin --no-ffmpeg-fallback --no-display --out-dir Phase_4_Network_Latency/runs/video_linux_ethernet_clean_rtsp_tcp
```

Network cleanup:

```bash
sudo tc qdisc del dev eth0 root
```

## video_linux_ethernet_clean_rtsp_udp

- family: `video_network`
- network: `clean`
- video: `rtsp_udp`
- alert: `none`

Network setup:

```bash
sudo tc qdisc del dev eth0 root
```

Run:

```powershell
python3 Phase_3_Fusion_MultiCam/run_live_campaign.py --versions V4 --models yolov8s --formats fp32_engine --cameras cam_02,cam_03,cam_05,cam_07 --duration-min 10 --device cuda:0 --record-fps 25 --object-min-camera-votes 2 --capture-backend gstreamer --gst-protocol udp --gst-latency-ms 50 --gst-pipeline decodebin --no-ffmpeg-fallback --no-display --out-dir Phase_4_Network_Latency/runs/video_linux_ethernet_clean_rtsp_udp
```

Network cleanup:

```bash
sudo tc qdisc del dev eth0 root
```

## video_linux_ethernet_moderate_rtsp_tcp

- family: `video_network`
- network: `moderate`
- video: `rtsp_tcp`
- alert: `none`

Network setup:

```bash
sudo tc qdisc replace dev eth0 root netem delay 80ms 25ms loss 1.0%
```

Run:

```powershell
python3 Phase_3_Fusion_MultiCam/run_live_campaign.py --versions V4 --models yolov8s --formats fp32_engine --cameras cam_02,cam_03,cam_05,cam_07 --duration-min 10 --device cuda:0 --record-fps 25 --object-min-camera-votes 2 --capture-backend gstreamer --gst-protocol tcp --gst-latency-ms 50 --gst-pipeline decodebin --no-ffmpeg-fallback --no-display --out-dir Phase_4_Network_Latency/runs/video_linux_ethernet_moderate_rtsp_tcp
```

Network cleanup:

```bash
sudo tc qdisc del dev eth0 root
```

## video_linux_ethernet_moderate_rtsp_udp

- family: `video_network`
- network: `moderate`
- video: `rtsp_udp`
- alert: `none`

Network setup:

```bash
sudo tc qdisc replace dev eth0 root netem delay 80ms 25ms loss 1.0%
```

Run:

```powershell
python3 Phase_3_Fusion_MultiCam/run_live_campaign.py --versions V4 --models yolov8s --formats fp32_engine --cameras cam_02,cam_03,cam_05,cam_07 --duration-min 10 --device cuda:0 --record-fps 25 --object-min-camera-votes 2 --capture-backend gstreamer --gst-protocol udp --gst-latency-ms 50 --gst-pipeline decodebin --no-ffmpeg-fallback --no-display --out-dir Phase_4_Network_Latency/runs/video_linux_ethernet_moderate_rtsp_udp
```

Network cleanup:

```bash
sudo tc qdisc del dev eth0 root
```

## video_linux_ethernet_severe_rtsp_tcp

- family: `video_network`
- network: `severe`
- video: `rtsp_tcp`
- alert: `none`

Network setup:

```bash
sudo tc qdisc replace dev eth0 root netem delay 200ms 75ms loss 5.0%
```

Run:

```powershell
python3 Phase_3_Fusion_MultiCam/run_live_campaign.py --versions V4 --models yolov8s --formats fp32_engine --cameras cam_02,cam_03,cam_05,cam_07 --duration-min 10 --device cuda:0 --record-fps 25 --object-min-camera-votes 2 --capture-backend gstreamer --gst-protocol tcp --gst-latency-ms 50 --gst-pipeline decodebin --no-ffmpeg-fallback --no-display --out-dir Phase_4_Network_Latency/runs/video_linux_ethernet_severe_rtsp_tcp
```

Network cleanup:

```bash
sudo tc qdisc del dev eth0 root
```

## video_linux_ethernet_severe_rtsp_udp

- family: `video_network`
- network: `severe`
- video: `rtsp_udp`
- alert: `none`

Network setup:

```bash
sudo tc qdisc replace dev eth0 root netem delay 200ms 75ms loss 5.0%
```

Run:

```powershell
python3 Phase_3_Fusion_MultiCam/run_live_campaign.py --versions V4 --models yolov8s --formats fp32_engine --cameras cam_02,cam_03,cam_05,cam_07 --duration-min 10 --device cuda:0 --record-fps 25 --object-min-camera-votes 2 --capture-backend gstreamer --gst-protocol udp --gst-latency-ms 50 --gst-pipeline decodebin --no-ffmpeg-fallback --no-display --out-dir Phase_4_Network_Latency/runs/video_linux_ethernet_severe_rtsp_udp
```

Network cleanup:

```bash
sudo tc qdisc del dev eth0 root
```

## alert_linux_ethernet_websocket

- family: `alert_transport`
- network: `clean`
- video: `rtsp_tcp`
- alert: `websocket`

Network setup:

```bash
sudo tc qdisc del dev eth0 root
```

Run:

```powershell
python3 Phase_4_Network_Latency/alert_delivery_benchmark.py --transport websocket --qos 0 --events 500 --rate-hz 25 --out Phase_4_Network_Latency/runs/alert_linux_ethernet_websocket/alert_latency.csv
```

Network cleanup:

```bash
sudo tc qdisc del dev eth0 root
```

## alert_linux_ethernet_mqtt_qos0

- family: `alert_transport`
- network: `clean`
- video: `rtsp_tcp`
- alert: `mqtt_qos0`

Network setup:

```bash
sudo tc qdisc del dev eth0 root
```

Run:

```powershell
python3 Phase_4_Network_Latency/alert_delivery_benchmark.py --transport mqtt --qos 0 --events 500 --rate-hz 25 --out Phase_4_Network_Latency/runs/alert_linux_ethernet_mqtt_qos0/alert_latency.csv
```

Network cleanup:

```bash
sudo tc qdisc del dev eth0 root
```

## alert_linux_ethernet_mqtt_qos1

- family: `alert_transport`
- network: `clean`
- video: `rtsp_tcp`
- alert: `mqtt_qos1`

Network setup:

```bash
sudo tc qdisc del dev eth0 root
```

Run:

```powershell
python3 Phase_4_Network_Latency/alert_delivery_benchmark.py --transport mqtt --qos 1 --events 500 --rate-hz 25 --out Phase_4_Network_Latency/runs/alert_linux_ethernet_mqtt_qos1/alert_latency.csv
```

Network cleanup:

```bash
sudo tc qdisc del dev eth0 root
```

## video_linux_wifi_clean_rtsp_tcp

- family: `video_network`
- network: `clean`
- video: `rtsp_tcp`
- alert: `none`

Network setup:

```bash
sudo tc qdisc del dev eth0 root
```

Run:

```powershell
python3 Phase_3_Fusion_MultiCam/run_live_campaign.py --versions V4 --models yolov8s --formats fp32_engine --cameras cam_02,cam_03,cam_05,cam_07 --duration-min 10 --device cuda:0 --record-fps 25 --object-min-camera-votes 2 --capture-backend gstreamer --gst-protocol tcp --gst-latency-ms 50 --gst-pipeline decodebin --no-ffmpeg-fallback --no-display --out-dir Phase_4_Network_Latency/runs/video_linux_wifi_clean_rtsp_tcp
```

Network cleanup:

```bash
sudo tc qdisc del dev eth0 root
```

## video_linux_wifi_clean_rtsp_udp

- family: `video_network`
- network: `clean`
- video: `rtsp_udp`
- alert: `none`

Network setup:

```bash
sudo tc qdisc del dev eth0 root
```

Run:

```powershell
python3 Phase_3_Fusion_MultiCam/run_live_campaign.py --versions V4 --models yolov8s --formats fp32_engine --cameras cam_02,cam_03,cam_05,cam_07 --duration-min 10 --device cuda:0 --record-fps 25 --object-min-camera-votes 2 --capture-backend gstreamer --gst-protocol udp --gst-latency-ms 50 --gst-pipeline decodebin --no-ffmpeg-fallback --no-display --out-dir Phase_4_Network_Latency/runs/video_linux_wifi_clean_rtsp_udp
```

Network cleanup:

```bash
sudo tc qdisc del dev eth0 root
```

## video_linux_wifi_moderate_rtsp_tcp

- family: `video_network`
- network: `moderate`
- video: `rtsp_tcp`
- alert: `none`

Network setup:

```bash
sudo tc qdisc replace dev eth0 root netem delay 80ms 25ms loss 1.0%
```

Run:

```powershell
python3 Phase_3_Fusion_MultiCam/run_live_campaign.py --versions V4 --models yolov8s --formats fp32_engine --cameras cam_02,cam_03,cam_05,cam_07 --duration-min 10 --device cuda:0 --record-fps 25 --object-min-camera-votes 2 --capture-backend gstreamer --gst-protocol tcp --gst-latency-ms 50 --gst-pipeline decodebin --no-ffmpeg-fallback --no-display --out-dir Phase_4_Network_Latency/runs/video_linux_wifi_moderate_rtsp_tcp
```

Network cleanup:

```bash
sudo tc qdisc del dev eth0 root
```

## video_linux_wifi_moderate_rtsp_udp

- family: `video_network`
- network: `moderate`
- video: `rtsp_udp`
- alert: `none`

Network setup:

```bash
sudo tc qdisc replace dev eth0 root netem delay 80ms 25ms loss 1.0%
```

Run:

```powershell
python3 Phase_3_Fusion_MultiCam/run_live_campaign.py --versions V4 --models yolov8s --formats fp32_engine --cameras cam_02,cam_03,cam_05,cam_07 --duration-min 10 --device cuda:0 --record-fps 25 --object-min-camera-votes 2 --capture-backend gstreamer --gst-protocol udp --gst-latency-ms 50 --gst-pipeline decodebin --no-ffmpeg-fallback --no-display --out-dir Phase_4_Network_Latency/runs/video_linux_wifi_moderate_rtsp_udp
```

Network cleanup:

```bash
sudo tc qdisc del dev eth0 root
```

## video_linux_wifi_severe_rtsp_tcp

- family: `video_network`
- network: `severe`
- video: `rtsp_tcp`
- alert: `none`

Network setup:

```bash
sudo tc qdisc replace dev eth0 root netem delay 200ms 75ms loss 5.0%
```

Run:

```powershell
python3 Phase_3_Fusion_MultiCam/run_live_campaign.py --versions V4 --models yolov8s --formats fp32_engine --cameras cam_02,cam_03,cam_05,cam_07 --duration-min 10 --device cuda:0 --record-fps 25 --object-min-camera-votes 2 --capture-backend gstreamer --gst-protocol tcp --gst-latency-ms 50 --gst-pipeline decodebin --no-ffmpeg-fallback --no-display --out-dir Phase_4_Network_Latency/runs/video_linux_wifi_severe_rtsp_tcp
```

Network cleanup:

```bash
sudo tc qdisc del dev eth0 root
```

## video_linux_wifi_severe_rtsp_udp

- family: `video_network`
- network: `severe`
- video: `rtsp_udp`
- alert: `none`

Network setup:

```bash
sudo tc qdisc replace dev eth0 root netem delay 200ms 75ms loss 5.0%
```

Run:

```powershell
python3 Phase_3_Fusion_MultiCam/run_live_campaign.py --versions V4 --models yolov8s --formats fp32_engine --cameras cam_02,cam_03,cam_05,cam_07 --duration-min 10 --device cuda:0 --record-fps 25 --object-min-camera-votes 2 --capture-backend gstreamer --gst-protocol udp --gst-latency-ms 50 --gst-pipeline decodebin --no-ffmpeg-fallback --no-display --out-dir Phase_4_Network_Latency/runs/video_linux_wifi_severe_rtsp_udp
```

Network cleanup:

```bash
sudo tc qdisc del dev eth0 root
```

## alert_linux_wifi_websocket

- family: `alert_transport`
- network: `clean`
- video: `rtsp_tcp`
- alert: `websocket`

Network setup:

```bash
sudo tc qdisc del dev eth0 root
```

Run:

```powershell
python3 Phase_4_Network_Latency/alert_delivery_benchmark.py --transport websocket --qos 0 --events 500 --rate-hz 25 --out Phase_4_Network_Latency/runs/alert_linux_wifi_websocket/alert_latency.csv
```

Network cleanup:

```bash
sudo tc qdisc del dev eth0 root
```

## alert_linux_wifi_mqtt_qos0

- family: `alert_transport`
- network: `clean`
- video: `rtsp_tcp`
- alert: `mqtt_qos0`

Network setup:

```bash
sudo tc qdisc del dev eth0 root
```

Run:

```powershell
python3 Phase_4_Network_Latency/alert_delivery_benchmark.py --transport mqtt --qos 0 --events 500 --rate-hz 25 --out Phase_4_Network_Latency/runs/alert_linux_wifi_mqtt_qos0/alert_latency.csv
```

Network cleanup:

```bash
sudo tc qdisc del dev eth0 root
```

## alert_linux_wifi_mqtt_qos1

- family: `alert_transport`
- network: `clean`
- video: `rtsp_tcp`
- alert: `mqtt_qos1`

Network setup:

```bash
sudo tc qdisc del dev eth0 root
```

Run:

```powershell
python3 Phase_4_Network_Latency/alert_delivery_benchmark.py --transport mqtt --qos 1 --events 500 --rate-hz 25 --out Phase_4_Network_Latency/runs/alert_linux_wifi_mqtt_qos1/alert_latency.csv
```

Network cleanup:

```bash
sudo tc qdisc del dev eth0 root
```
