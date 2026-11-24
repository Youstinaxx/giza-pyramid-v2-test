// "How the pyramid was built" overlay: stone courses laid block by block, a golden capstone, then fireworks launched
// from behind the pyramid. Pure three.js, no model files. Everything is in pyramid units: base = 1 (centred on 0,0),
// y up, apex at y = H. The page gives it the camera every frame, so it stays on the pyramid like the ring does.
(function () {
  const COURSES = 24, BUILD_S = 16, CAP_S = 1.4, FIRE_START = 1.2, FIRE_S = 9;

  function make(THREE) {
    const group = new THREE.Group(); group.name = 'pyramid-build';
    let H = 0, built = null, t0 = null, fw = null;

    // ---------- stones ----------
    const stoneMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.92, metalness: 0, vertexColors: false });
    const goldMat = new THREE.MeshStandardMaterial({ color: 0xffcf5a, emissive: 0x7a4b00, emissiveIntensity: 0.8, metalness: 0.85, roughness: 0.25 });
    const SHOW_VEIL = false;   // pale cover over the unbuilt part of the real pyramid: turned off
    const veilMat = new THREE.MeshBasicMaterial({ color: 0xaebfcf, transparent: true, opacity: 0.42, depthWrite: false, side: THREE.DoubleSide });
    const veilPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);   // the veil covers only the part not built yet
    veilMat.clippingPlanes = [veilPlane];

    function buildStones(h) {
      if (built) { group.remove(built.blocks, built.slabs, built.cap, built.veil, built.glow); built.blocks.geometry.dispose(); built.slabs.geometry.dispose(); }
      H = h; const hc = H / COURSES, bw = hc * 1.45, blocks = [], slabs = [];
      for (let i = 0; i < COURSES - 1; i++) {                       // the last course is the capstone
        const s = 1 - i / COURSES, n = Math.max(1, Math.round(s / bw)), w = s / n, y = i * hc + hc / 2;
        const ring = [];
        if (n === 1) ring.push([0, 0]);
        else for (let k = 0; k < n - 1; k++) {                      // walk the four edges of the course
          const a = -s / 2 + w / 2 + k * w;
          ring.push([a, -s / 2 + w / 2], [s / 2 - w / 2, a], [-a, s / 2 - w / 2], [-s / 2 + w / 2, -a]);
        }
        ring.forEach(([x, z]) => blocks.push({ i, x, y, z, w, d: w }));
        const inner = s - 2 * w; if (inner > 0) slabs.push({ i, y, s: inner });
      }
      // timing: equal time per block, so the wide bottom courses take longest (as they would have)
      blocks.forEach((b, k) => { b.start = k / blocks.length * BUILD_S; });
      const firstOf = {}; blocks.forEach(b => { if (firstOf[b.i] === undefined) firstOf[b.i] = b.start; });
      const lastOf = {}; blocks.forEach(b => { lastOf[b.i] = b.start; });
      slabs.forEach(sl => { sl.start = firstOf[sl.i]; sl.end = lastOf[sl.i] + 0.5; });

      const geo = new THREE.BoxGeometry(1, 1, 1);
      const inst = new THREE.InstancedMesh(geo, stoneMat, blocks.length); inst.renderOrder = 1;
      const col = new THREE.Color(), rnd = mulberry(7);
      blocks.forEach((b, k) => { const v = 0.82 + 0.16 * rnd(); col.setRGB(0.93 * v, 0.83 * v, 0.64 * v); inst.setColorAt(k, col); b.jx = (rnd() - 0.5) * 0.06; b.jz = (rnd() - 0.5) * 0.06; });
      inst.instanceColor.needsUpdate = true;
      const slabInst = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1), new THREE.MeshStandardMaterial({ color: 0xc9b58a, roughness: 0.95 }), Math.max(1, slabs.length));
      slabInst.renderOrder = 1;
      // golden capstone (pyramidion): a 4-sided cone turned 45° so its faces line up with the pyramid's
      const yc = (COURSES - 1) * hc, sc = 1 / COURSES;
      const cap = new THREE.Mesh(new THREE.ConeGeometry(sc / Math.SQRT2, H - yc, 4, 1), goldMat);
      cap.rotation.y = Math.PI / 4; cap.renderOrder = 1; cap.userData.rest = yc + (H - yc) / 2;
      const glow = new THREE.PointLight(0xffd27a, 0, 0.6); glow.position.set(0, H, 0);
      // veil: a pale truncated-free pyramid shell over the real pyramid, clipped to what isn't built yet
      const veilGeo = new THREE.ConeGeometry(Math.SQRT1_2 * 1.004, H * 1.004, 4, 1, true); veilGeo.translate(0, H * 1.004 / 2, 0);
      const veil = new THREE.Mesh(veilGeo, veilMat); veil.rotation.y = Math.PI / 4; veil.renderOrder = 2;
      group.add(inst, slabInst, cap, glow, veil);
      built = { blocks: inst, slabs: slabInst, cap, glow, veil, list: blocks, slabList: slabs, hc };
    }

    // ---------- fireworks ----------
    // Drawn solid (not light-adding) so they show against a bright daytime sky: each spark = a soft disc with its own
    // colour, opacity and size, plus a short streak behind it.
    const MAX = 9000;
    const pGeo = new THREE.BufferGeometry();
    const pPos = new Float32Array(MAX * 3), pCol = new Float32Array(MAX * 3), pA = new Float32Array(MAX), pS = new Float32Array(MAX);
    pGeo.setAttribute('position', new THREE.BufferAttribute(pPos, 3)); pGeo.setAttribute('color', new THREE.BufferAttribute(pCol, 3));
    pGeo.setAttribute('alpha', new THREE.BufferAttribute(pA, 1)); pGeo.setAttribute('psize', new THREE.BufferAttribute(pS, 1));
    const pMat = new THREE.ShaderMaterial({
      uniforms: { scale: { value: 400 } }, transparent: true, depthWrite: false,
      vertexShader: 'attribute vec3 color; attribute float alpha; attribute float psize; uniform float scale; varying vec3 vC; varying float vA;' +
        'void main(){ vC = color; vA = alpha; vec4 mv = modelViewMatrix * vec4(position, 1.0); gl_PointSize = max(1.5, psize * scale / -mv.z); gl_Position = projectionMatrix * mv; }',
      fragmentShader: 'varying vec3 vC; varying float vA; void main(){ float d = length(gl_PointCoord - 0.5) * 2.0; if (d > 1.0) discard;' +
        ' float core = smoothstep(0.55, 0.0, d); vec3 c = mix(vC, vec3(1.0), core * 0.65); gl_FragColor = vec4(c, vA * smoothstep(1.0, 0.35, d)); }',
    });
    const points = new THREE.Points(pGeo, pMat); points.frustumCulled = false; points.renderOrder = 3; group.add(points);
    const PALETTE = [[1, 0.78, 0.15], [1, 0.25, 0.35], [0.2, 0.75, 1], [0.95, 0.35, 1], [0.3, 1, 0.45], [1, 0.5, 0.1], [1, 1, 1]];

    function fireworksPlan(camC) {
      // launch from the ground behind the pyramid (the side away from the camera); rockets rise hidden, burst above
      const d = Math.hypot(camC[0], camC[2]) || 1, bx = -camC[0] / d, bz = -camC[2] / d, px = -bz, pz = bx, rnd = mulberry(11), rockets = [];
      let t = 0;
      while (t < FIRE_S - 2) { rockets.push(rocket(t)); t += 0.35 + 0.35 * rnd(); }
      for (let k = 0; k < 6; k++) rockets.push(rocket(FIRE_S - 1.8 + k * 0.08));      // finale
      function rocket(at) {
        const side = (rnd() - 0.5) * 1.3, back = 0.45 + 0.3 * rnd();
        return { at, x: bx * back + px * side, z: bz * back + pz * side, top: H * (1.35 + 0.7 * rnd()), rise: 1.1 + 0.4 * rnd(),
                 col: PALETTE[Math.floor(rnd() * PALETTE.length)], col2: PALETTE[Math.floor(rnd() * PALETTE.length)], n: 110 + Math.floor(rnd() * 60), seed: Math.floor(rnd() * 1e9),
                 speed: H * (0.9 + 0.5 * rnd()) };
      }
      return rockets;
    }

    function drawFireworks(t) {   // t = seconds since fireworks began; fully determined by t (no state to drift)
      let n = 0;
      const put = (x, y, z, c, a, size) => { if (n >= MAX) return; pPos[3 * n] = x; pPos[3 * n + 1] = y; pPos[3 * n + 2] = z; pCol[3 * n] = c[0]; pCol[3 * n + 1] = c[1]; pCol[3 * n + 2] = c[2]; pA[n] = a; pS[n] = size; n++; };
      const S0 = H * 0.05;                                          // spark size, in pyramid units
      for (const r of fw) {
        const tt = t - r.at; if (tt < 0) continue;
        const ease = e => r.top * (1 - (1 - e) * (1 - e));
        if (tt < r.rise) {                                           // rocket climbing, with a spark trail
          const e = tt / r.rise;
          for (let k = 0; k < 10; k++) { const ek = Math.max(0, e - k * 0.02); put(r.x, ease(ek), r.z, [1, 0.85, 0.5], 1 - k / 10, S0 * (1 - k / 14)); }
          continue;
        }
        const tb = tt - r.rise, life = 2.2; if (tb > life) continue;
        const rnd = mulberry(r.seed), fade = Math.pow(1 - tb / life, 1.3);
        const at = tb2 => (1 - Math.exp(-2.4 * tb2)) / 2.4;          // distance travelled (slowing down)
        for (let k = 0; k < r.n; k++) {                              // burst: sparks on a sphere, slowing, then falling
          const u = rnd() * 2 - 1, a = rnd() * Math.PI * 2, sq = Math.sqrt(1 - u * u), sp = r.speed * (0.8 + 0.3 * rnd());
          const c = rnd() < 0.55 ? r.col : r.col2, tw = 0.8 + 0.2 * Math.sin(tb * 25 + k);
          for (let s2 = 0; s2 < 3; s2++) {                           // the spark + a short streak behind it
            const tb2 = Math.max(0, tb - s2 * 0.06), d = sp * at(tb2), y = r.top + u * d - 0.12 * H * tb2 * tb2;
            put(r.x + sq * Math.cos(a) * d, y, r.z + sq * Math.sin(a) * d, c, fade * tw * (1 - s2 * 0.35), S0 * (1.1 - s2 * 0.3) * (0.6 + 0.4 * fade));
          }
        }
        if (tb < 0.15) put(r.x, r.top, r.z, [1, 1, 0.9], 1 - tb / 0.15, S0 * 6 * (0.5 + tb * 4));   // flash
      }
      pGeo.setDrawRange(0, n); for (const k of ['position', 'color', 'alpha', 'psize']) pGeo.attributes[k].needsUpdate = true;
    }

    const m4 = new THREE.Matrix4(), q = new THREE.Quaternion(), v = new THREE.Vector3(), sc3 = new THREE.Vector3();
    const easeBack = x => { const c = 1.4; return 1 + (c + 1) * Math.pow(x - 1, 3) + c * Math.pow(x - 1, 2); };

    return {
      group,
      get started() { return t0 !== null; },
      reset() { t0 = null; fw = null; },
      // h = pyramid height in base units; camC = camera position in pyramid units; now = seconds
      update(h, camC, now, viewHeightPx, projScale) {
        if (viewHeightPx) pMat.uniforms.scale.value = projScale * viewHeightPx / 2;   // spark sizes in world units
        if (!built || Math.abs(h - H) > 1e-6) buildStones(h);
        if (t0 === null) t0 = now;
        const t = now - t0, B = built, FLY = 0.55;
        // blocks: each falls into its place from above, slightly outside, with a small settle
        B.list.forEach((b, k) => {
          const p = Math.max(0, Math.min(1, (t - b.start) / FLY));
          if (p <= 0) { m4.makeScale(0, 0, 0); B.blocks.setMatrixAt(k, m4); return; }
          const e = easeBack(p), lift = (1 - e) * B.hc * 5;
          v.set(b.x + b.jx * (1 - e) * 0, b.y + lift, b.z); sc3.set(b.w * 0.985, B.hc * 0.97, b.d * 0.985).multiplyScalar(Math.min(1, 0.4 + 0.6 * p));
          m4.compose(v, q.identity(), sc3); B.blocks.setMatrixAt(k, m4);
        });
        B.blocks.instanceMatrix.needsUpdate = true;
        B.slabList.forEach((s, k) => {
          const p = Math.max(0, Math.min(1, (t - s.start) / Math.max(0.1, s.end - s.start)));
          m4.compose(v.set(0, s.y - B.hc / 2 + B.hc * p / 2, 0), q.identity(), sc3.set(s.s, Math.max(1e-4, B.hc * p), s.s)); B.slabs.setMatrixAt(k, m4);
        });
        B.slabs.instanceMatrix.needsUpdate = true;
        // capstone: lowered onto the top once the last course is done, then a flash of light
        const tc = t - BUILD_S - 0.3, pc = Math.max(0, Math.min(1, tc / CAP_S));
        B.cap.visible = tc > 0; B.cap.position.y = B.cap.userData.rest + (1 - easeBack(pc)) * H * 0.6;
        B.glow.intensity = tc > CAP_S ? 3 * Math.exp(-(tc - CAP_S) * 1.5) : 0;
        // the veil hides the real pyramid above the courses laid so far, so it looks as if it isn't there yet
        const builtTo = Math.min(H, B.list.reduce((m, b) => (t - b.start > FLY ? Math.max(m, b.y + B.hc / 2) : m), 0));
        veilPlane.constant = -(pc >= 1 ? H * 2 : builtTo); B.veil.visible = SHOW_VEIL && pc < 1;
        // fireworks from behind, after the capstone
        const tf = t - BUILD_S - 0.3 - CAP_S - FIRE_START;
        if (tf >= 0) { if (!fw) fw = fireworksPlan(camC); drawFireworks(tf); points.visible = tf < FIRE_S + 2.5; } else { points.visible = false; fw = null; }
        return { t, phase: t < BUILD_S ? 'building' : pc < 1 ? 'capstone' : tf < FIRE_S + 2.5 ? 'fireworks' : 'done', builtTo };
      },
    };
  }
  function mulberry(a) { return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
  const api = { make, COURSES, BUILD_S };
  if (typeof module !== 'undefined') module.exports = api; else self.PyramidBuild = api;
})();
