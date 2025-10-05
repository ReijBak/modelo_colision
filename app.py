import math
import io
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, render_template_string, jsonify, send_file
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

app = Flask(__name__)

# =======================================================
# CLASE PRINCIPAL (estructura limpia sin self.)
# =======================================================
class ModeloImpactoAsteroide:
    def __init__(self,
                 mu,
                 R_e,
                 I0_pos,
                 V_mov,
                 d_ast,
                 r_ast,
                 S_time,
                 S_Step,
                 In_angle,
                 vH0,
                 vV0):

        # --- Parámetros auxiliares internos ---
        nframes_riesgo = 35
        lat_step_deg = 5.0
        lon_step_deg = 5.0

        # --- Simulación completa orquestada desde aquí ---
        positions, times = ModeloImpactoAsteroide.simular_trayectoria(
            mu, R_e, I0_pos, V_mov, S_time, S_Step
        )

        lats, lons, corredor, impacto = ModeloImpactoAsteroide.proyectar_a_tierra(positions)

        riesgo_frames = ModeloImpactoAsteroide.generar_frames_riesgo(
            impacto, nframes_riesgo, lat_step_deg, lon_step_deg
        )

        app2 = ModeloImpactoAsteroide.crear_app(
            positions, times, lats, lons, corredor, impacto, riesgo_frames, R_e, nframes_riesgo
        )

        app2.run(host="127.0.0.1", port=5000, debug=False)

    # =======================================================
    def ecuaciones_movimiento(mu, r, v):
        rnorm = np.linalg.norm(r) + 1e-18
        a = -mu * r / (rnorm ** 3)
        return v, a

    # =======================================================
    def simular_trayectoria(mu, r_earth_m, r0_m, v0_ms, duracion_horas, dt_s):
        t_max_s = duracion_horas * 3600.0
        max_steps = int(t_max_s / dt_s)
        r = np.array(r0_m, dtype=float)
        v = np.array(v0_ms, dtype=float)
        positions, times = [], []
        t0 = datetime.utcnow()

        for step in range(max_steps):
            positions.append(r.copy())
            times.append(t0 + timedelta(seconds=step * dt_s))
            dr, a = ModeloImpactoAsteroide.ecuaciones_movimiento(mu, r, v)
            v += a * dt_s
            r += dr * dt_s
            if np.linalg.norm(r) <= r_earth_m:
                positions.append(r.copy())
                times.append(t0 + timedelta(seconds=(step + 1) * dt_s))
                break
        return np.array(positions), np.array(times)

    # =======================================================
    def proyectar_a_tierra(positions):
        rnorm = np.linalg.norm(positions, axis=1) + 1e-18
        lats = np.degrees(np.arcsin(positions[:, 2] / rnorm))
        lons = np.degrees(np.arctan2(positions[:, 1], positions[:, 0]))

        rng = np.random.default_rng(42)
        ruido = rng.normal(0, 1.5, size=(len(lats), 2))
        corredor = np.vstack([lats + ruido[:, 0], lons + ruido[:, 1]]).T
        impacto = (float(lats[-1]), float(lons[-1]))
        return lats, lons, corredor, impacto

    # =======================================================
    def generar_frames_riesgo(impacto, nframes_riesgo, lat_step_deg, lon_step_deg):
        frames = []
        lat_edges = np.arange(-90, 90, lat_step_deg)
        lon_edges = np.arange(-180, 180, lon_step_deg)
        lat_centers = lat_edges + 0.5 * lat_step_deg
        lon_centers = lon_edges + 0.5 * lon_step_deg

        for idx in range(nframes_riesgo):
            t = idx / (nframes_riesgo - 1)
            baseline = 80.0 * (1 - t)
            sigma = 60.0 * (1 - t) + 3.0
            peak = 100.0 * t
            frame_cells = []
            for latc in lat_centers:
                for lonc in lon_centers:
                    dlat = latc - impacto[0]
                    dlon = (lonc - impacto[1] + 180) % 360 - 180
                    dist = math.sqrt(dlat ** 2 + dlon ** 2)
                    bump = peak * math.exp(-(dist ** 2) / (2 * sigma ** 2))
                    prob = baseline + bump
                    prob = float(np.clip(prob, 0, 100))
                    frame_cells.append({
                        "west": lonc - 0.5 * lon_step_deg,
                        "south": latc - 0.5 * lat_step_deg,
                        "east": lonc + 0.5 * lon_step_deg,
                        "north": latc + 0.5 * lat_step_deg,
                        "prob": prob
                    })
            frames.append(frame_cells)
        return frames

    # =======================================================
    def crear_app(positions, times, lats, lons, corredor, impacto, riesgo_frames, r_earth_m, nframes_riesgo):

        @app.route("/")
        def index():
            return render_template_string(ModeloImpactoAsteroide.html_template(), nframes=nframes_riesgo)

        @app.route("/trayectoria")
        def trayectoria():
            rnorm = np.linalg.norm(positions, axis=1)
            alts = rnorm - r_earth_m
            return jsonify({
                "positions": [
                    {"lat": float(lat), "lon": float(lon), "alt": float(max(0, alt))}
                    for lat, lon, alt in zip(
                        np.degrees(np.arcsin(positions[:, 2] / rnorm)),
                        np.degrees(np.arctan2(positions[:, 1], positions[:, 0])),
                        alts
                    )
                ],
                "times": [t.isoformat() for t in times]
            })

        @app.route("/riesgo/<int:f>")
        def riesgo(f):
            return jsonify(riesgo_frames[f % nframes_riesgo])

        @app.route("/map2d.png")
        def map2d():
            buf = ModeloImpactoAsteroide.make_2d_map_png(lats, lons, corredor, impacto, times)
            return send_file(buf, mimetype="image/png")

        return app

    # =======================================================
    def make_2d_map_png(lats, lons, corredor, impacto, times, width=1400, height=800, padding_deg=30):
        fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_title("Trayectoria final y zona probable de impacto (95%)", fontsize=14)
        ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.6)
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.6)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4)

        ax.plot(lons, lats, linestyle='--', color='red', linewidth=2, label='Órbita nominal (proyección)')
        ax.scatter(corredor[:, 1], corredor[:, 0], s=15, color='orange', alpha=0.35, label='Corredor 95%')
        ax.scatter(impacto[1], impacto[0], color='red', s=250, alpha=0.7, marker='o', label='Zona de impacto')

        min_lat = float(min(lats.min(), impacto[0]) - padding_deg)
        max_lat = float(max(lats.max(), impacto[0]) + padding_deg)
        min_lon = float(min(lons.min(), impacto[1]) - padding_deg)
        max_lon = float(max(lons.max(), impacto[1]) + padding_deg)
        ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

        nlab = min(6, len(times))
        for k in np.linspace(0, len(times) - 1, nlab, dtype=int):
            ax.text(lons[k], lats[k], times[k].strftime("%H:%M UTC"), fontsize=8, color='black', transform=ccrs.PlateCarree())

        ax.legend(loc='lower left')
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        return buf

    # =======================================================
    def html_template():
        return """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Simulación de impacto - Cesium</title>
  <script src="https://cesium.com/downloads/cesiumjs/releases/1.121/Build/Cesium/Cesium.js"></script>
  <link href="https://cesium.com/downloads/cesiumjs/releases/1.121/Build/Cesium/Widgets/widgets.css" rel="stylesheet">
  <style>
    html,body,#cesiumContainer{width:100%;height:100%;margin:0;padding:0;overflow:hidden;}
    .panel{position:absolute;left:10px;top:10px;background:rgba(255,255,255,0.9);padding:8px;border-radius:6px;font-family:Arial;z-index:999;}
    .panel_right{position:absolute;right:10px;top:10px;background:rgba(255,255,255,0.9);padding:8px;border-radius:6px;font-family:Arial;z-index:999;}
    .btn{margin-top:6px;padding:6px 8px;background:#0077cc;color:white;border:none;border-radius:4px;cursor:pointer;}
  </style>
</head>
<body>
<div id="cesiumContainer"></div>
<div class="panel">
  <div><b>Base:</b></div>
  <select id="baseSelect">
    <option value="osm">OpenStreetMap</option>
    <option value="esri">Esri Satellite</option>
  </select>
  <div style="margin-top:6px;"><a href="/map2d.png" target="_blank">🌍 Abrir mapa 2D</a></div>
</div>
<div class="panel_right">
  <div><b>Evolución del riesgo</b></div>
  <div>Frame: <span id="frameNum">0</span></div>
  <div>UTC: <span id="utcLabel">-</span></div>
  <button id="playBtn" class="btn">▶ Play</button>
  <button id="pauseBtn" class="btn" style="background:#777">⏸ Pause</button>
</div>

<script>
const viewer = new Cesium.Viewer("cesiumContainer", {
  imageryProvider: new Cesium.UrlTemplateImageryProvider({url:"https://a.tile.openstreetmap.org/{z}/{x}/{y}.png"}),
  baseLayerPicker:false,animation:false,timeline:false
});

document.getElementById("baseSelect").addEventListener("change",e=>{
  viewer.imageryLayers.removeAll();
  if(e.target.value==="esri"){
    viewer.imageryLayers.addImageryProvider(new Cesium.UrlTemplateImageryProvider({url:"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"}));
  }else{
    viewer.imageryLayers.addImageryProvider(new Cesium.UrlTemplateImageryProvider({url:"https://a.tile.openstreetmap.org/{z}/{x}/{y}.png"}));
  }
});

const NFRAMES={{nframes}};
let frame=0;let playing=true;let overlayEntities=[];let traj=[],trajTimes=[];let asteroid=null;let trajEntity=null;

function probColor(p){
  if(p>=80)return Cesium.Color.RED.withAlpha(0.5);
  if(p>=60)return Cesium.Color.ORANGE.withAlpha(0.4);
  if(p>=40)return Cesium.Color.YELLOW.withAlpha(0.3);
  if(p>=20)return Cesium.Color.LIME.withAlpha(0.25);
  return Cesium.Color.DARKGREEN.withAlpha(0.2);
}

async function loadTrajectory(){
  const r=await fetch("/trayectoria");
  const d=await r.json();
  traj=d.positions;trajTimes=d.times;
  const pos=traj.map(p=>Cesium.Cartesian3.fromDegrees(p.lon,p.lat,p.alt));
  trajEntity=viewer.entities.add({polyline:{positions:pos,width:3,material:Cesium.Color.RED.withAlpha(0.6)}});
  asteroid=viewer.entities.add({position:pos[0],point:{pixelSize:10,color:Cesium.Color.CRIMSON}});
  viewer.zoomTo(trajEntity);
}

async function drawFrame(f){
  const r=await fetch("/riesgo/"+f);
  const cells=await r.json();
  for(let e of overlayEntities)viewer.entities.remove(e);
  overlayEntities=[];
  for(let c of cells){
    const rect=Cesium.Rectangle.fromDegrees(c.west,c.south,c.east,c.north);
    overlayEntities.push(viewer.entities.add({rectangle:{coordinates:rect,material:probColor(c.prob),outline:false}}));
  }
  const idx=Math.min(traj.length-1,Math.floor(f/(NFRAMES-1)*(traj.length-1)));
  const p=traj[idx];
  asteroid.position=Cesium.Cartesian3.fromDegrees(p.lon,p.lat,p.alt);
  document.getElementById("frameNum").innerText=f+1;
  const t=new Date(trajTimes[idx]);
  document.getElementById("utcLabel").innerText=t.toISOString().replace('T',' ').split('.')[0]+" UTC";
}

document.getElementById("playBtn").onclick=()=>playing=true;
document.getElementById("pauseBtn").onclick=()=>playing=false;

(async function(){
  await loadTrajectory();
  drawFrame(0);
  setInterval(()=>{if(playing){drawFrame(frame);frame=(frame+1)%NFRAMES;}},500);
})();
</script>
</body></html>"""


# =======================================================
# MAIN
# =======================================================
if __name__ == "__main__":
    print("Inicializando simulación de impacto...")

    mu =3.986004418e14 # parámetro gravitacional estándar de la Tierra
    R_e = 6371000.0# Radio de la Tierra en [m]
    I0_pos = [0.0, -7.5e6, 4.0e6]  # posición inicial
    V_mov = [2500.0, 4500.0, -7200.0] # Velocidad  [m/s]
    d_ast = 3000.0 # densidad [kg/m^3]
    r_ast = 50.0 # radio [m]
    S_time = 1.0 # duración de la simulación [horas]
    S_Step = 2.0 # paso temporal de la simulación [segundos]
    In_angle = 35 # angulo de entrada [°]
    vH0 = 5.0 # velocidad horizontal inicial, lateral [Km/s]
    vV0 = 5.0 # velocidad vertical inicial, cae [Km/s]

    ModeloImpactoAsteroide(mu, R_e, I0_pos, V_mov, d_ast, r_ast, S_time, S_Step, In_angle, vH0, vV0)
    
