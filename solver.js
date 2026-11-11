/* PyramidSolver — camera pose relative to a square pyramid from 5 view-dependent keypoints.
 *
 * World (pyramid) frame: origin = centre of the base, Y up, base edges parallel to X and Z,
 * base = 1 unit, apex at (0, H, 0). Camera frame (OpenCV): x right, y down, z forward.
 * Keypoints (image px): A apex, L leftmost base corner, R rightmost, N nearest, F farthest,
 * each [u, v, conf]. Because the pyramid has 4-fold symmetry, yaw is only defined mod 90°.
 *
 * solve(kp, {f, cx, cy, up?, init?}) -> {R, t, rvec, C, err, ok}
 *   up: optional unit "world up" vector in camera coordinates (from the phone tilt sensor)
 *   init: previous result (warm start)
 */
(function (root) {
  const H_DEFAULT = 0.6365;                       // Khufu original height / base
  const CORNERS = [[-.5, 0, -.5], [.5, 0, -.5], [.5, 0, .5], [-.5, 0, .5]];

  function rodrigues(r) {
    const th = Math.hypot(r[0], r[1], r[2]);
    if (th < 1e-12) return [1, 0, 0, 0, 1, 0, 0, 0, 1];
    const k = [r[0] / th, r[1] / th, r[2] / th], c = Math.cos(th), s = Math.sin(th), v = 1 - c;
    return [c + k[0] * k[0] * v, k[0] * k[1] * v - k[2] * s, k[0] * k[2] * v + k[1] * s,
            k[1] * k[0] * v + k[2] * s, c + k[1] * k[1] * v, k[1] * k[2] * v - k[0] * s,
            k[2] * k[0] * v - k[1] * s, k[2] * k[1] * v + k[0] * s, c + k[2] * k[2] * v];
  }
  function logmap(R) {
    const c = Math.max(-1, Math.min(1, (R[0] + R[4] + R[8] - 1) / 2)), th = Math.acos(c);
    if (th < 1e-9) return [0, 0, 0];
    if (Math.PI - th < 1e-4) { // near 180°: axis from diagonal
      const x = Math.sqrt(Math.max(0, (R[0] + 1) / 2)), y = Math.sqrt(Math.max(0, (R[4] + 1) / 2)) * (R[1] >= 0 ? 1 : -1), z = Math.sqrt(Math.max(0, (R[8] + 1) / 2)) * (R[2] >= 0 ? 1 : -1);
      return [x * th, y * th, z * th];
    }
    const s = 2 * Math.sin(th);
    return [(R[7] - R[5]) / s * th, (R[2] - R[6]) / s * th, (R[3] - R[1]) / s * th];
  }
  const mv = (R, p) => [R[0] * p[0] + R[1] * p[1] + R[2] * p[2], R[3] * p[0] + R[4] * p[1] + R[5] * p[2], R[6] * p[0] + R[7] * p[1] + R[8] * p[2]];
  const mtv = (R, p) => [R[0] * p[0] + R[3] * p[1] + R[6] * p[2], R[1] * p[0] + R[4] * p[1] + R[7] * p[2], R[2] * p[0] + R[5] * p[1] + R[8] * p[2]];
  const norm = v => { const n = Math.hypot(...v) || 1; return v.map(x => x / n); };
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

  function lookAt(C, T) {                          // OpenCV world->camera rotation
    const f = norm([T[0] - C[0], T[1] - C[1], T[2] - C[2]]);
    const r = norm(cross(f, [0, 1, 0])), d = cross(f, r);
    return [r[0], r[1], r[2], d[0], d[1], d[2], f[0], f[1], f[2]];
  }

  function project(p, X, cam) {
    const R = rodrigues(p), Xc = mv(R, X); Xc[0] += p[3]; Xc[1] += p[4]; Xc[2] += p[5];
    return { u: cam.f * Xc[0] / Xc[2] + cam.cx, v: cam.f * Xc[1] / Xc[2] + cam.cy, z: Xc[2] };
  }

  // which model corner each view-dependent label refers to, for a given pose
  function labelCorners(p, cam) {
    const P = CORNERS.map(c => project(p, c, cam));
    const iu = P.map((q, i) => i).sort((a, b) => P[a].u - P[b].u), iz = P.map((q, i) => i).sort((a, b) => P[a].z - P[b].z);
    return { P, L: iu[0], R: iu[3], N: iz[0], F: iz[3] };
  }

  function residuals(p, kp, cam, opt) {
    const H = opt.H, out = [];
    const a = project(p, [0, H, 0], cam);
    if (a.z <= 0) return null;
    const w0 = kp[0][2];
    out.push((a.u - kp[0][0]) * w0, (a.v - kp[0][1]) * w0);
    const lab = labelCorners(p, cam);
    if (lab.P.some(q => q.z <= 0)) return null;
    ['L', 'R', 'N', 'F'].forEach((name, j) => {
      const k = kp[j + 1];
      if (!k || k[2] < opt.minConf) { out.push(0, 0); return; }
      const q = lab.P[lab[name]], w = k[2] * (name === 'F' ? 0.5 : 1);  // far corner is always hidden: trust less
      out.push((q.u - k[0]) * w, (q.v - k[1]) * w);
    });
    if (opt.up) {                                  // camera's view of world-up must match the tilt sensor
      const g = mv(rodrigues(p), [0, 1, 0]), s = cam.f * opt.gravityWeight;
      out.push((g[0] - opt.up[0]) * s, (g[1] - opt.up[1]) * s, (g[2] - opt.up[2]) * s);
    }
    const C = mtv(rodrigues(p), [-p[3], -p[4], -p[5]]);
    out.push(C[1] < 0 ? -C[1] * cam.f : 0);        // stay above the ground
    return out;
  }

  function solve6(A, b) {                          // Gaussian elimination, 6x6
    const n = 6, M = A.map((r, i) => r.concat([b[i]]));
    for (let i = 0; i < n; i++) {
      let m = i; for (let k = i + 1; k < n; k++) if (Math.abs(M[k][i]) > Math.abs(M[m][i])) m = k;
      [M[i], M[m]] = [M[m], M[i]]; if (Math.abs(M[i][i]) < 1e-15) return null;
      for (let k = i + 1; k < n; k++) { const f = M[k][i] / M[i][i]; for (let j = i; j <= n; j++) M[k][j] -= f * M[i][j]; }
    }
    const x = new Array(n).fill(0);
    for (let i = n - 1; i >= 0; i--) { let s = M[i][n]; for (let j = i + 1; j < n; j++) s -= M[i][j] * x[j]; x[i] = s / M[i][i]; }
    return x;
  }
  const sse = r => r.reduce((s, x) => s + x * x, 0);

  function lm(p0, kp, cam, opt, iters) {
    let p = p0.slice(), r = residuals(p, kp, cam, opt); if (!r) return null;
    let e = sse(r), lam = 1e-3;
    for (let it = 0; it < iters; it++) {
      const J = [];
      for (let j = 0; j < 6; j++) {
        const h = j < 3 ? 1e-6 : 1e-6 * Math.max(1, Math.abs(p[5]));
        const q = p.slice(); q[j] += h; const rq = residuals(q, kp, cam, opt);
        if (!rq) return { p, e };
        J.push(rq.map((x, i) => (x - r[i]) / h));
      }
      const JTJ = [...Array(6)].map((_, a) => [...Array(6)].map((_, b) => J[a].reduce((s, x, i) => s + x * J[b][i], 0)));
      const JTr = [...Array(6)].map((_, a) => J[a].reduce((s, x, i) => s + x * r[i], 0));
      let improved = false;
      for (let tries = 0; tries < 8; tries++) {
        const A = JTJ.map((row, i) => row.map((v, j) => i === j ? v * (1 + lam) + 1e-9 : v));
        const d = solve6(A, JTr.map(x => -x)); if (!d) { lam *= 10; continue; }
        const q = p.map((x, i) => x + d[i]), rq = residuals(q, kp, cam, opt);
        if (rq) { const eq = sse(rq); if (eq < e) { p = q; r = rq; lam = Math.max(1e-7, lam / 5); improved = e - eq > 1e-10 * e; e = eq; break; } }
        lam *= 10;
      }
      if (!improved) break;
    }
    return { p, e };
  }

  function initPose(kp, cam, H, yaw, box) {
    // distance from apparent height: pyramid height ≈ H units over (base y - apex y) pixels
    const baseV = box ? box[3] : Math.max(...kp.slice(1).filter(k => k && k[2] > 0.3).map(k => k[1]).concat([kp[0][1] + 50]));
    const hpx = Math.max(20, baseV - kp[0][1]);
    const d = Math.max(1.2, cam.f * H / hpx);
    const C = [d * Math.sin(yaw), 0.01, d * Math.cos(yaw)];
    // aim so the apex lands near its detected pixel: tilt the look-at target by the pixel offset
    const T = [0, H * 0.5, 0];
    const R = lookAt(C, T), t = mv(R, C).map(x => -x), r = logmap(R);
    return [r[0], r[1], r[2], t[0], t[1], t[2]];
  }

  function solve(kp, cam, o = {}) {
    const opt = { H: o.H || H_DEFAULT, up: o.up ? norm(o.up) : null, gravityWeight: o.gravityWeight ?? 0.3, minConf: o.minConf ?? 0.5 };
    if (!kp[0] || kp[0][2] < 0.5) return { ok: false, reason: 'apex not found' };
    const starts = [];
    if (o.init && o.init.p) starts.push(o.init.p);
    if (!o.init || !o.init.p || o.forceSearch) for (let k = 0; k < 6; k++) starts.push(initPose(kp, cam, opt.H, (k * 15 + 7.5) * Math.PI / 180, o.box));
    let best = null;
    for (const s of starts) {
      const res = lm(s, kp, cam, opt, 40);
      if (res && (!best || res.e < best.e)) best = res;
    }
    if (!best) return { ok: false, reason: 'no solution' };
    const used = 1 + kp.slice(1).filter(k => k && k[2] >= opt.minConf).length;
    const R = rodrigues(best.p), t = best.p.slice(3), C = mtv(R, t.map(x => -x));
    const rp = residuals(best.p, kp, cam, { ...opt, up: null });
    const px = Math.sqrt(sse(rp) / Math.max(1, used));       // RMS reprojection error per point (px, conf-weighted)
    return { ok: px < 0.06 * cam.f, p: best.p, R, t, C, err: px, used, H: opt.H };
  }

  // edges for drawing: [i, j] over vertices 0..3 corners, 4 apex; plus face visibility from camera centre
  function wireframe(sol, cam, opts = {}) {
    const H = sol.H, V = CORNERS.concat([[0, H, 0]]);
    const P = V.map(X => project(sol.p, X, cam));
    const faces = [0, 1, 2, 3].map(i => {
      const a = V[i], b = V[(i + 1) % 4], n = norm(cross([b[0] - a[0], 0, b[2] - a[2]], [-a[0], H, -a[2]]));
      const m = [(a[0] + b[0]) / 3, H / 3, (a[2] + b[2]) / 3];
      return (sol.C[0] - m[0]) * n[0] + (sol.C[1] - m[1]) * n[1] + (sol.C[2] - m[2]) * n[2] > 0;
    });
    const edges = [];
    for (let i = 0; i < 4; i++) {
      edges.push({ a: P[i], b: P[(i + 1) % 4], visible: faces[i] });                    // base edge i belongs to face i
      edges.push({ a: P[i], b: P[4], visible: faces[i] || faces[(i + 3) % 4] });      // ridge between faces i-1 and i
    }
    // "latitude" lines (constant height) and "longitude" lines (base point -> apex) on visible faces.
    // opts.levels: height fractions 0..1; opts.lons: fractions along each base edge 0..1
    const grid = [];
    const levels = opts.levels || [0.25, 0.5, 0.75], lons = opts.lons || [0.25, 0.5, 0.75];
    for (let f = 0; f < 4; f++) {
      if (!faces[f]) continue;
      const a = V[f], b = V[(f + 1) % 4], ap = V[4];
      for (const s of levels) {
        const p1 = a.map((x, k) => x + (ap[k] - x) * s), p2 = b.map((x, k) => x + (ap[k] - x) * s);
        const q1 = project(sol.p, p1, cam), q2 = project(sol.p, p2, cam);
        grid.push(Object.assign([q1, q2], { kind: 'lat', frac: s, face: f }));
      }
      for (const s of lons) {
        const p = a.map((x, k) => x + (b[k] - x) * s);
        grid.push(Object.assign([project(sol.p, p, cam), P[4]], { kind: 'lon', frac: s, face: f }));
      }
    }
    return { P, edges, grid, faces };
  }

  // camera pose for three.js (camera looks down -Z, Y up), in pyramid units
  function threePose(sol) {
    const Rt = [sol.R[0], sol.R[3], sol.R[6], sol.R[1], sol.R[4], sol.R[7], sol.R[2], sol.R[5], sol.R[8]]; // camera->world
    const M = [Rt[0], -Rt[1], -Rt[2], Rt[3], -Rt[4], -Rt[5], Rt[6], -Rt[7], -Rt[8]];                      // flip y,z
    const tr = M[0] + M[4] + M[8]; let q;
    if (tr > 0) { const s = Math.sqrt(tr + 1) * 2; q = [(M[7] - M[5]) / s, (M[2] - M[6]) / s, (M[3] - M[1]) / s, s / 4]; }
    else if (M[0] > M[4] && M[0] > M[8]) { const s = Math.sqrt(1 + M[0] - M[4] - M[8]) * 2; q = [s / 4, (M[1] + M[3]) / s, (M[2] + M[6]) / s, (M[7] - M[5]) / s]; }
    else if (M[4] > M[8]) { const s = Math.sqrt(1 + M[4] - M[0] - M[8]) * 2; q = [(M[1] + M[3]) / s, s / 4, (M[5] + M[7]) / s, (M[2] - M[6]) / s]; }
    else { const s = Math.sqrt(1 + M[8] - M[0] - M[4]) * 2; q = [(M[2] + M[6]) / s, (M[5] + M[7]) / s, s / 4, (M[3] - M[1]) / s]; }
    return { position: sol.C, quaternion: q };
  }

  // world-up in camera coords from DeviceOrientation (beta, gamma in degrees) and screen angle
  function upFromOrientation(beta, gamma, screenAngle) {
    const b = beta * Math.PI / 180, g = gamma * Math.PI / 180;
    let d = [-Math.cos(b) * Math.sin(g), Math.sin(b), Math.cos(b) * Math.cos(g)];        // device coords: x right, y top, z out of screen
    const a = -(screenAngle || 0) * Math.PI / 180;                                     // rotate into the screen's current x/y
    d = [d[0] * Math.cos(a) - d[1] * Math.sin(a), d[0] * Math.sin(a) + d[1] * Math.cos(a), d[2]];
    return [d[0], -d[1], -d[2]];                                                        // back camera, OpenCV axes
  }

  const api = { solve, wireframe, threePose, upFromOrientation, project, rodrigues, lookAt, logmap, labelCorners, CORNERS, H_DEFAULT };
  if (typeof module !== 'undefined') module.exports = api; else root.PyramidSolver = api;
})(typeof self !== 'undefined' ? self : this);
