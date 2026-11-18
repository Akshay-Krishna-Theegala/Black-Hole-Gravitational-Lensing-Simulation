import time
import numpy as np
from vispy import app, gloo

# =====================================================================
# 1. STARFIELD BACKGROUND SHADER WITH TEST STAR (CLEAN NO-GLOW)
# =====================================================================
BG_VERT = """
attribute vec2 a_position;
varying vec2 v_uv;
void main() {
    v_uv = a_position;
    gl_Position = vec4(a_position, 0.999, 1.0);
}
"""

BG_FRAG = """
varying vec2 v_uv;
uniform vec2 u_resolution;
uniform vec3 u_cam_pos;
uniform vec3 u_cam_forward;
uniform vec3 u_cam_right;
uniform vec3 u_cam_up;
uniform float u_mass;
uniform float u_star_thresh;
uniform float u_fov;

vec3 hash33(vec3 p3) {
    p3 = fract(p3 * vec3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yxz + 33.33);
    return fract((p3.xxy + p3.yxx) * p3.zyx);
}

void main() {
    float aspect = u_resolution.x / u_resolution.y;
    vec2 uv = v_uv;
    uv.x *= aspect;

    vec3 rayDir = normalize(u_cam_forward * u_fov + u_cam_right * uv.x + u_cam_up * uv.y);
    float rs = u_mass;

    float proj = dot(-u_cam_pos, rayDir);
    vec3 closestPoint = u_cam_pos + rayDir * max(0.0, proj);
    float b = length(closestPoint);

    float deflection = (1.5 * rs * rs) / (b * b + 0.04);
    vec3 toCenter = normalize(-u_cam_pos);
    vec3 radialDir = normalize(cross(cross(rayDir, toCenter), rayDir));
    vec3 lensedDir = normalize(rayDir + radialDir * deflection * 0.35);

    vec3 col = vec3(0.0);

    // 1. Procedural Background Starfield
    for (int layer = 0; layer < 2; layer++) {
        float scale = (layer == 0) ? 90.0 : 165.0;
        vec3 p = lensedDir * scale;
        vec3 id = floor(p);
        vec3 f = fract(p) - 0.5;
        vec3 rnd = hash33(id + float(layer) * 19.3);

        if (rnd.x > u_star_thresh) {
            float d = length(f - (rnd - 0.5) * 0.5);
            float span = max(0.001, 1.0 - u_star_thresh);
            float brightness = smoothstep(0.36, 0.02, d) * ((rnd.x - u_star_thresh) / span);
            vec3 starTint = mix(vec3(1.0, 0.72, 0.48), vec3(0.68, 0.85, 1.0), rnd.y);
            col += starTint * brightness * (layer == 0 ? 1.4 : 0.7);
        }
    }

    // 2. THE TEST STAR
    vec3 testStarWorldPos = vec3(0.15, 0.25, -4.0); 
    vec3 dirToTestStar = normalize(testStarWorldPos);
    float starDist = length(lensedDir - dirToTestStar);
    float testStarGlow = smoothstep(0.08, 0.0, starDist);
    col += vec3(0.0, 1.0, 1.0) * testStarGlow * 5.0;

    col = col / (col + vec3(0.8));
    gl_FragColor = vec4(pow(col, vec3(0.85)), 1.0);
}
"""

# =====================================================================
# 2. SOLID 3D EVENT HORIZON SPHERE SHADER
# =====================================================================
SPHERE_VERT = """
attribute vec3 a_position;
uniform vec2 u_resolution;
uniform vec3 u_cam_pos;
uniform vec3 u_cam_forward;
uniform vec3 u_cam_right;
uniform vec3 u_cam_up;
uniform float u_fov;
uniform float u_mass;

void main() {
    float rs = u_mass;
    vec3 worldPos = a_position * rs;

    vec3 toPart = worldPos - u_cam_pos;
    float distToCam = length(toPart);
    vec3 dirToPart = toPart / distToCam;

    float zCam = dot(dirToPart, u_cam_forward);
    if (zCam <= 0.05) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        return;
    }

    float aspect = u_resolution.x / u_resolution.y;
    float xNdc = (dot(dirToPart, u_cam_right) / zCam) * (u_fov / aspect);
    float yNdc = (dot(dirToPart, u_cam_up) / zCam) * u_fov;
    float depthNdc = clamp((distToCam / 30.0) * 2.0 - 1.0, -0.99, 0.99);

    gl_Position = vec4(xNdc, yNdc, depthNdc, 1.0);
}
"""

SPHERE_FRAG = """
void main() {
    gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
}
"""

# =====================================================================
# 3. ACCURATE GRAVITATIONALLY LENSED ACCRETION DISK SHADER (REDSHIFT & FADE)
# =====================================================================
PARTICLE_VERT = """
attribute float a_radius;       
attribute float a_angle0;       
attribute float a_y_offset;     
attribute float a_size;         
attribute float a_temp;         
attribute float a_image_mode;   

uniform vec2 u_resolution;
uniform float u_time;
uniform vec3 u_cam_pos;
uniform vec3 u_cam_forward;
uniform vec3 u_cam_right;
uniform vec3 u_cam_up;
uniform float u_mass;
uniform float u_fov;

varying vec3 v_color;
varying float v_alpha;

void main() {
    float rs = u_mass;
    float r = a_radius * rs;

    float omega = 2.6 / pow(max(0.35, a_radius), 1.35);
    float angle = a_angle0 + u_time * omega;

    vec3 worldPos = vec3(
        r * cos(angle),
        a_y_offset * rs,
        r * sin(angle)
    );

    vec3 camPos = u_cam_pos;
    vec3 v_cam_to_bh = normalize(-camPos);

    vec3 toPart = worldPos - camPos;
    float distToCam = length(toPart);
    vec3 dirToPart = toPart / distToCam;

    float alongBhAxis = dot(worldPos, v_cam_to_bh);
    vec3 perpVec = worldPos - v_cam_to_bh * alongBhAxis;
    float impactParam = length(perpVec);

    float shadowR = rs * 1.0; 
    vec3 lensedDir = dirToPart;
    v_alpha = 1.0;

    if (alongBhAxis > 0.0) {
        vec3 radialUp = (impactParam > 1e-4) ? (perpVec / impactParam) : u_cam_up;
        float einsteinR = sqrt(1.3 * rs * alongBhAxis);

        if (a_image_mode < 0.5) {
            float lensedImpact = 0.5 * (impactParam + sqrt(impactParam * impactParam + 4.0 * einsteinR * einsteinR));
            float behindFactor = smoothstep(0.0, rs * 1.5, alongBhAxis);
            float finalImpact = mix(impactParam, max(lensedImpact, shadowR * 1.0), behindFactor);

            vec3 lensedWorld = v_cam_to_bh * alongBhAxis + radialUp * finalImpact;
            lensedDir = normalize(lensedWorld - camPos);
        } else {
            if (impactParam > rs * 3.5) {
                v_alpha = 0.0;
            } else {
                float secImpact = 0.5 * (-impactParam + sqrt(impactParam * impactParam + 4.0 * einsteinR * einsteinR));
                secImpact = max(secImpact, shadowR * 1.01);
                vec3 lensedWorld = v_cam_to_bh * alongBhAxis - radialUp * secImpact;
                lensedDir = normalize(lensedWorld - camPos);
                v_alpha = smoothstep(rs * 3.5, rs * 0.2, impactParam) * 0.85;
            }
        }
    } else {
        if (a_image_mode > 0.5) {
            v_alpha = 0.0;
        }
    }

    float zCam = dot(lensedDir, u_cam_forward);
    if (zCam <= 0.05 || v_alpha <= 0.01) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        gl_PointSize = 0.0;
        return;
    }

    float aspect = u_resolution.x / u_resolution.y;
    float xNdc = (dot(lensedDir, u_cam_right) / zCam) * (u_fov / aspect);
    float yNdc = (dot(lensedDir, u_cam_up) / zCam) * u_fov;

    float depthNdc = clamp((distToCam / 35.0) * 2.0 - 1.0, -0.95, 0.95);
    gl_Position = vec4(xNdc, yNdc, depthNdc, 1.0);

    float perspectiveScale = (u_resolution.y * u_fov * 0.5) / max(0.5, distToCam);
    gl_PointSize = clamp(a_size * rs * perspectiveScale, 2.0, 45.0);

    vec3 velDir = normalize(vec3(-sin(angle), 0.0, cos(angle)));
    float doppler = clamp(1.0 + dot(velDir, -dirToPart) * 0.55, 0.45, 1.75);

    // --- GRAVITATIONAL REDSHIFT & HORIZON FADE-OUT ---
    float horizonProximity = clamp((a_radius - 1.0) / 0.5, 0.0, 1.0); 
    vec3 redshiftCol = vec3(0.75, 0.1, 0.0); // Deep red as it falls inward
    vec3 normalCol   = mix(vec3(0.95, 0.3, 0.0), vec3(1.0, 0.85, 0.7), a_temp);
    vec3 baseCol     = mix(redshiftCol, normalCol, horizonProximity);

    v_alpha *= smoothstep(0.0, 0.3, horizonProximity);
    // ------------------------------------------------

    v_color = baseCol * doppler;
}
"""

PARTICLE_FRAG = """
varying vec3 v_color;
varying float v_alpha;

void main() {
    vec2 coord = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(coord, coord);
    if (r2 > 1.0) {
        discard;
    }

    float z = sqrt(1.0 - r2);
    vec3 normal = vec3(coord.x, -coord.y, z);
    float ndotl = max(0.25, dot(normal, normalize(vec3(0.3, 0.5, 0.8))));
    float coreGlow = pow(z, 0.65);
    float rimGlow = pow(1.0 - z, 2.0) * 0.35;

    vec3 sphereColor = v_color * (ndotl * 0.85 + coreGlow * 0.65) + v_color * rimGlow;
    float alpha = smoothstep(1.0, 0.72, sqrt(r2)) * v_alpha;

    gl_FragColor = vec4(sphereColor, alpha);
}
"""


class BlackHoleParticleSim(app.Canvas):
    def __init__(self, num_spheres=80000):
        super().__init__(keys='interactive', size=(1200, 780), vsync=True)
        self.num_spheres = num_spheres

        self.bg_program = gloo.Program(BG_VERT, BG_FRAG)
        self.bg_program['a_position'] = np.array(
            [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)], dtype=np.float32
        )

        self.sphere_program = gloo.Program(SPHERE_VERT, SPHERE_FRAG)
        self._build_sphere_mesh(stacks=32, slices=32)

        self.particle_program = gloo.Program(PARTICLE_VERT, PARTICLE_FRAG)
        self.reset_state()
        self._build_accretion_spheres()

        self.last_frame_time = time.time()
        self._timer = app.Timer('auto', connect=self.on_timer, start=True)
        self.show()

    def _build_sphere_mesh(self, stacks=32, slices=32):
        positions = []
        for i in range(stacks + 1):
            lat = np.pi * (-0.5 + float(i) / stacks)
            sin_lat, cos_lat = np.sin(lat), np.cos(lat)
            for j in range(slices + 1):
                lon = 2.0 * np.pi * float(j) / slices
                x = cos_lat * np.cos(lon)
                y = sin_lat
                z = cos_lat * np.sin(lon)
                positions.append([x, y, z])

        indices = []
        for i in range(stacks):
            for j in range(slices):
                p1 = i * (slices + 1) + j
                p2 = p1 + slices + 1
                indices.extend([p1, p2, p1 + 1, p2, p2 + 1, p1 + 1])

        verts = np.array(positions, dtype=np.float32)
        inds = np.array(indices, dtype=np.uint32)

        self.sphere_indices = gloo.IndexBuffer(inds)
        self.sphere_program['a_position'] = gloo.VertexBuffer(verts)

    def _build_accretion_spheres(self):
        rng = np.random.default_rng(42)
        n = self.num_spheres

        u = rng.random(n).astype(np.float32)

        radii = self.disk_inner_radius + (u ** 1.35) * (self.disk_width - self.disk_inner_radius)

        arm_offset = (rng.integers(0, 3, size=n) * (2.0 * np.pi / 3.0)).astype(np.float32)
        angles = arm_offset + radii * 1.15 + rng.normal(0.0, 0.45, size=n).astype(np.float32)

        thickness = (self.disk_thickness + 0.02 * (radii - self.disk_inner_radius)).astype(np.float32)
        y_offsets = rng.normal(0.0, thickness).astype(np.float32)

        sizes = ((0.035 + (rng.random(n) ** 2.6) * 0.095) * self.particle_size).astype(np.float32)

        temps = np.clip(
            1.0 - (radii - self.disk_inner_radius) / (self.disk_width - self.disk_inner_radius) + rng.normal(0.0, 0.08,
                                                                                                             n), 0.0,
            1.0).astype(np.float32)

        all_radii = np.concatenate([radii, radii])
        all_angles = np.concatenate([angles, angles])
        all_y = np.concatenate([y_offsets, y_offsets])
        all_sizes = np.concatenate([sizes, sizes * 0.85])
        all_temps = np.concatenate([temps, temps])
        all_modes = np.concatenate([
            np.zeros(n, dtype=np.float32),
            np.ones(n, dtype=np.float32),
        ])

        self.particle_program['a_radius'] = all_radii
        self.particle_program['a_angle0'] = all_angles
        self.particle_program['a_y_offset'] = all_y
        self.particle_program['a_size'] = all_sizes
        self.particle_program['a_temp'] = all_temps
        self.particle_program['a_image_mode'] = all_modes

    def reset_state(self):
        self.yaw = 0.35
        self.pitch = 0.24
        self.distance = 10.5
        self.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.mass = 1.0
        self.show_disk = True
        self.star_thresh = 0.96
        self.fov = 1.4
        self.spin_speed = 1.0
        self.paused = False
        self.sim_time = 0.0

        self.disk_thickness = 0.02
        self.disk_width = 6.0
        self.disk_inner_radius = 1.0
        self.particle_size = 1.0

        self._update_title()

    def _update_title(self):
        disk_str = "ON" if self.show_disk else "OFF"
        pause_str = " [PAUSED]" if self.paused else ""
        density_pct = int(round((1.0 - self.star_thresh) * 1000))
        self.title = (
            f"3D Accurate Black Hole{pause_str} | Mass: {self.mass:.2f} (+/-) | "
            f"Size: {self.particle_size:.2f}x (p/P) | Thickness: {self.disk_thickness} | "
            f"Spheres: {disk_str} (D) | Spin: {self.spin_speed:.1f}x (Up/Dn) | Reset (R)"
        )

    def _get_camera_basis(self):
        cx = self.distance * np.cos(self.pitch) * np.sin(self.yaw)
        cy = self.distance * np.sin(self.pitch)
        cz = self.distance * np.cos(self.pitch) * np.cos(self.yaw)
        offset = np.array([cx, cy, cz], dtype=np.float32)
        cam_pos = self.target + offset

        forward = -offset / np.linalg.norm(offset)
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        return cam_pos, forward, right, up

    def on_timer(self, event):
        now = time.time()
        dt = now - self.last_frame_time
        self.last_frame_time = now
        if not self.paused:
            self.sim_time += dt * self.spin_speed
        self.update()

    def on_draw(self, event):
        gloo.clear(color='black', depth=True)
        gloo.set_viewport(0, 0, *self.size)

        cam_pos, forward, right, up = self._get_camera_basis()

        gloo.set_state(depth_test=False, blend=False)
        self.bg_program['u_resolution'] = self.size
        self.bg_program['u_cam_pos'] = cam_pos
        self.bg_program['u_cam_forward'] = forward
        self.bg_program['u_cam_right'] = right
        self.bg_program['u_cam_up'] = up
        self.bg_program['u_mass'] = self.mass
        self.bg_program['u_star_thresh'] = self.star_thresh
        self.bg_program['u_fov'] = self.fov
        self.bg_program.draw('triangle_strip')

        gloo.set_state(depth_test=True, depth_func='less', blend=False)
        self.sphere_program['u_resolution'] = self.size
        self.sphere_program['u_cam_pos'] = cam_pos
        self.sphere_program['u_cam_forward'] = forward
        self.sphere_program['u_cam_right'] = right
        self.sphere_program['u_cam_up'] = up
        self.sphere_program['u_fov'] = self.fov
        self.sphere_program['u_mass'] = self.mass
        self.sphere_program.draw('triangles', self.sphere_indices)

        if self.show_disk:
            gloo.set_state(
                depth_test=True,
                depth_func='lequal',
                blend=True,
                blend_func=('src_alpha', 'one'),
            )
            self.particle_program['u_resolution'] = self.size
            self.particle_program['u_time'] = self.sim_time
            self.particle_program['u_cam_pos'] = cam_pos
            self.particle_program['u_cam_forward'] = forward
            self.particle_program['u_cam_right'] = right
            self.particle_program['u_cam_up'] = up
            self.particle_program['u_mass'] = self.mass
            self.particle_program['u_fov'] = self.fov
            self.particle_program.draw('points')

    def on_mouse_move(self, event):
        if event.is_dragging and event.last_event is not None:
            dx, dy = event.pos - event.last_event.pos
            if 1 in event.buttons:
                self.yaw -= dx * 0.007
                self.pitch = np.clip(self.pitch + dy * 0.007, -1.45, 1.45)
            elif 2 in event.buttons:
                _, forward, right, up = self._get_camera_basis()
                pan_scale = self.distance * 0.0015
                self.target -= right * (dx * pan_scale) - up * (dy * pan_scale)

    def on_mouse_wheel(self, event):
        min_dist = max(4.5, self.mass * 3.5)
        self.distance = np.clip(self.distance - event.delta[1] * 0.6, min_dist, 25.0)

    def on_key_press(self, event):
        key = event.key.name if event.key is not None else ''
        text = event.text

        if text in ('d', 'D'):
            self.show_disk = not self.show_disk
        elif text in ('+', '='):
            self.mass = min(2.4, self.mass + 0.1)
            self.distance = max(self.distance, self.mass * 3.5)
        elif text in ('-', '_'):
            self.mass = max(0.3, self.mass - 0.1)
        elif text == '[':
            self.star_thresh = min(0.992, self.star_thresh + 0.008)
        elif text == ']':
            self.star_thresh = max(0.88, self.star_thresh - 0.008)
        elif text == 'p':
            self.particle_size = max(0.1, self.particle_size - 0.05)
            self._build_accretion_spheres()
        elif text == 'P':
            self.particle_size = min(3.0, self.particle_size + 0.05)
            self._build_accretion_spheres()
        elif key == 'Up':
            self.spin_speed = min(5.0, self.spin_speed + 0.2)
        elif key == 'Down':
            self.spin_speed = max(-5.0, self.spin_speed - 0.2)
        elif key == 'Space':
            self.paused = not self.paused
        elif text in ('r', 'R'):
            self.reset_state()
            self._build_accretion_spheres()

        self._update_title()


if __name__ == '__main__':
    canvas = BlackHoleParticleSim(num_spheres=800000)

    # Custom disk configuration
    canvas.disk_thickness = 0.02
    canvas.disk_width = 6.0
    canvas.disk_inner_radius = 0.9
    canvas.particle_size = 0.5 # Standard scale for nice distinct particle spheres

    canvas._build_accretion_spheres()
    canvas._update_title()

    app.run()
