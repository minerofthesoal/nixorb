import QtQuick 2.15

// NixOrb floating orb — "Siri 2.0" redesign.
//
// The brief: match the look of the iOS Dynamic Island's Siri glow — a dark
// glass surface with a soft, shifting, iridescent gradient (gold → pink →
// purple → blue) wrapping its edge, not a single flat-shaded sphere.
//
// That look is inherently a moving gradient, so unlike the previous design
// (a Canvas painted once per colour change, animated only via cheap Item
// transforms) the glow layer here repaints every tick. At the orb's actual
// size — under 200px even at the largest configured orb_size — a handful of
// radial gradients per frame is not a meaningful cost, and it's the only way
// to get the organic, liquid drift the reference look depends on. A Timer
// drives repaints at a fixed 30fps rather than the display's own refresh
// rate, so the cost doesn't scale with monitor Hz.
//
// Bridge properties (see nixorb/ui/orb_window.py — unchanged by this
// redesign): orbBridge.state / .amplitude / .color / .opacity / .orbSize.
// `color` (STATE_COLORS) still drives the thin rim accent, most visibly for
// "error" — the one state that must read as alarming rather than pretty —
// everything else about the palette is defined locally below.

Item {
    id: root
    width: orbBridge ? orbBridge.orbSize : 88
    height: width

    property string orbState: orbBridge ? orbBridge.state : "idle"
    property real orbAmplitude: orbBridge ? orbBridge.amplitude : 0.0
    property color orbColor: orbBridge ? orbBridge.color : "#C96442"
    property real orbOpacity: orbBridge ? orbBridge.opacity : 1.0

    readonly property real bodyRadius: width * 0.32
    readonly property real cx: width / 2
    readonly property real cy: height / 2

    opacity: orbOpacity
    antialiasing: true

    // ── Colour helpers ───────────────────────────────────────────── //
    function toColor(c) {
        return (typeof c === "string") ? Qt.color(c) : c
    }
    function css(c, a) {
        var col = toColor(c)
        return "rgba(" + Math.round(col.r * 255) + ","
                       + Math.round(col.g * 255) + ","
                       + Math.round(col.b * 255) + "," + a + ")"
    }
    function shade(c, amount) {
        var col = toColor(c)
        var t = Math.abs(amount)
        var to = amount > 0 ? 1.0 : 0.0
        return Qt.rgba(col.r + (to - col.r) * t,
                       col.g + (to - col.g) * t,
                       col.b + (to - col.b) * t, 1.0)
    }

    // ── Palette + motion per state ───────────────────────────────── //
    // Vivid, saturated hues — deliberately not STATE_COLORS' muted clay
    // palette, since a hue-shift of a muted colour just gives muddier
    // muted colours, not the vivid Apple-style spectrum this is chasing.
    function paletteFor(state) {
        switch (state) {
        case "listening": return ["#30D5C8", "#5AC8FA", "#0A84FF", "#63E6E2"]
        case "thinking":  return ["#FFD60A", "#FF9F0A", "#FF6482", "#BF5AF2"]
        case "speaking":  return ["#FF9F0A", "#FF375F", "#BF5AF2", "#5E5CE6", "#0AD1FF"]
        case "error":     return ["#FF6961", "#FF453A", "#B3453E"]
        default:          return ["#0A84FF", "#5E5CE6", "#5AC8FA", "#63E6E2"]  // idle
        }
    }
    readonly property var blobHues: paletteFor(orbState)

    // How fast the blobs swirl. Amplitude (voice level) nudges it further
    // on top of this, so the glow visibly livens up while you're talking.
    readonly property real baseSpeed: {
        if (orbState === "thinking") return 0.030
        if (orbState === "speaking") return 0.034
        if (orbState === "listening") return 0.020
        if (orbState === "error") return 0.010
        return 0.007  // idle — slow ambient drift, never fully still
    }

    property real t: 0
    Timer {
        interval: 33  // ~30fps — deliberately decoupled from display Hz
        running: true
        repeat: true
        onTriggered: {
            root.t += root.baseSpeed * (1.0 + root.orbAmplitude * 0.6)
            glow.requestPaint()
        }
    }

    // Gentle breathing scale, same cue the previous design used, kept for
    // continuity: a bit more presence while actively listening/speaking.
    property real pulse: 1.0
    SequentialAnimation on pulse {
        loops: Animation.Infinite
        running: orbState === "speaking" || orbState === "listening"
        NumberAnimation { to: 1.045; duration: 520; easing.type: Easing.InOutSine }
        NumberAnimation { to: 1.000; duration: 520; easing.type: Easing.InOutSine }
    }
    Behavior on pulse {
        enabled: !(orbState === "speaking" || orbState === "listening")
        NumberAnimation { duration: 240; easing.type: Easing.OutQuad }
    }

    Behavior on orbColor { ColorAnimation { duration: 260 } }

    // ── Outer bloom ──────────────────────────────────────────────── //
    // Cheap, GPU-composited ambient glow beyond the canvas edge — tinted
    // by the palette's lead hue rather than the flat orbColor, so the
    // bloom matches the iridescent body instead of clashing with it.
    Rectangle {
        anchors.centerIn: parent
        width: root.bodyRadius * 2 * 1.66 * root.pulse
        height: width
        radius: width / 2
        color: root.blobHues[0]
        opacity: (0.05 + root.orbAmplitude * 0.09)
        antialiasing: true
        Behavior on color { ColorAnimation { duration: 400 } }
    }
    Rectangle {
        anchors.centerIn: parent
        width: root.bodyRadius * 2 * 1.32 * root.pulse
        height: width
        radius: width / 2
        color: root.blobHues[root.blobHues.length > 1 ? 1 : 0]
        opacity: (0.09 + root.orbAmplitude * 0.14)
        antialiasing: true
        Behavior on opacity { NumberAnimation { duration: 140 } }
        Behavior on color { ColorAnimation { duration: 400 } }
    }

    // ── Iridescent glass body ────────────────────────────────────── //
    Canvas {
        id: glow
        anchors.fill: parent
        antialiasing: true
        smooth: true
        renderStrategy: Canvas.Cooperative
        scale: root.pulse

        onPaint: {
            var ctx = getContext("2d")
            ctx.reset()

            var cx = root.cx, cy = root.cy, r = root.bodyRadius
            var amp = root.orbAmplitude
            var pal = root.blobHues

            // Dark glass base — this is what makes the colours read as a
            // glowing surface rather than a flat rainbow disc; everything
            // saturated sits on top of it, additively, like light through
            // smoked glass.
            ctx.save()
            ctx.beginPath()
            ctx.arc(cx, cy, r, 0, Math.PI * 2)
            ctx.clip()

            var base = ctx.createRadialGradient(cx, cy, 0, cx, cy, r * 1.05)
            base.addColorStop(0.0, "rgba(24,24,32,0.94)")
            base.addColorStop(0.7, "rgba(14,14,20,0.96)")
            base.addColorStop(1.0, "rgba(6,6,10,0.98)")
            ctx.fillStyle = base
            ctx.fillRect(0, 0, width, height)

            // Ambient wash — a dim, full-disc tint of the palette before any
            // blob is drawn, so no frame of the animation ever shows a dead
            // black gap between blobs the way early passes of this did. The
            // blobs on top are what give it motion and bright highlights;
            // this just guarantees a floor of colour everywhere.
            ctx.globalCompositeOperation = "lighter"
            var n = pal.length
            for (var w = 0; w < n; w++) {
                var wash = ctx.createRadialGradient(cx, cy, 0, cx, cy, r * 1.02)
                wash.addColorStop(0.0, root.css(pal[w], 0.0))
                wash.addColorStop(0.55, root.css(pal[w], 0.018 + amp * 0.015))
                wash.addColorStop(1.0, root.css(pal[w], 0.06 + amp * 0.04))
                ctx.fillStyle = wash
                ctx.fillRect(0, 0, width, height)
            }

            // Iridescent blobs, orbiting near the rim on individual
            // lissajous-ish paths so they drift rather than spin in
            // lockstep. Additive blending is what produces the bright
            // colour mixing where two blobs overlap.
            for (var i = 0; i < n; i++) {
                var phase = root.t * (1.0 + i * 0.07) + (i * (Math.PI * 2 / n))
                var wobble = Math.sin(root.t * 1.3 + i * 1.7) * 0.08
                var orbitR = r * (0.62 + wobble + amp * 0.10)
                var bx = cx + Math.cos(phase) * orbitR
                var by = cy + Math.sin(phase * 0.82 + i) * orbitR
                var blobR = r * (0.56 + amp * 0.20)

                var g = ctx.createRadialGradient(bx, by, 0, bx, by, blobR)
                g.addColorStop(0.00, root.css(pal[i], 0.80))
                g.addColorStop(0.45, root.css(pal[i], 0.30))
                g.addColorStop(1.00, root.css(pal[i], 0.0))
                ctx.fillStyle = g
                ctx.beginPath()
                ctx.arc(bx, by, blobR, 0, Math.PI * 2)
                ctx.fill()
            }
            ctx.globalCompositeOperation = "source-over"

            // Glass specular highlight — the "liquid" cue.
            ctx.save()
            ctx.translate(cx - r * 0.28, cy - r * 0.38)
            ctx.rotate(-0.5)
            ctx.scale(1.0, 0.55)
            var spec = ctx.createRadialGradient(0, 0, 0, 0, 0, r * 0.58)
            spec.addColorStop(0.00, "rgba(255,255,255,0.55)")
            spec.addColorStop(0.45, "rgba(255,255,255,0.14)")
            spec.addColorStop(1.00, "rgba(255,255,255,0.0)")
            ctx.fillStyle = spec
            ctx.beginPath()
            ctx.arc(0, 0, r * 0.58, 0, Math.PI * 2)
            ctx.fill()
            ctx.restore()

            // Subtle inner shadow toward the bottom so the sphere still
            // reads as a rounded volume, not a lit-up flat disc.
            var shadow = ctx.createRadialGradient(
                cx, cy + r * 0.55, 0, cx, cy + r * 0.55, r * 0.9)
            shadow.addColorStop(0.0, "rgba(0,0,0,0.30)")
            shadow.addColorStop(1.0, "rgba(0,0,0,0.0)")
            ctx.fillStyle = shadow
            ctx.fillRect(0, 0, width, height)

            ctx.restore()  // undo clip

            // Rim: a soft glass edge normally, or a sharper, saturated
            // ring in the state colour for "error" — the one state that
            // needs to visibly interrupt the pretty gradient rather than
            // join it.
            ctx.beginPath()
            ctx.arc(cx, cy, r - 0.75, 0, Math.PI * 2)
            if (root.orbState === "error") {
                ctx.lineWidth = 2.2
                ctx.strokeStyle = root.css(root.orbColor, 0.9)
            } else {
                ctx.lineWidth = 1.1
                ctx.strokeStyle = "rgba(255,255,255,0.22)"
            }
            ctx.stroke()
        }
    }

    // Voice level brightens the glass from within — unchanged in spirit
    // from the previous design, just layered on top of the new body.
    Canvas {
        id: sheen
        anchors.fill: parent
        antialiasing: true
        smooth: true
        renderStrategy: Canvas.Cooperative
        scale: root.pulse
        opacity: root.orbAmplitude * 0.38
        visible: opacity > 0.01
        Behavior on opacity { NumberAnimation { duration: 90 } }

        onPaint: {
            var ctx = getContext("2d")
            ctx.reset()
            var r = root.bodyRadius
            var hx = root.cx - r * 0.16
            var hy = root.cy - r * 0.20
            var g = ctx.createRadialGradient(hx, hy, 0, hx, hy, r * 0.98)
            g.addColorStop(0.00, "rgba(255,255,255,0.85)")
            g.addColorStop(0.45, "rgba(255,255,255,0.28)")
            g.addColorStop(1.00, "rgba(255,255,255,0.0)")
            ctx.beginPath()
            ctx.arc(root.cx, root.cy, r, 0, Math.PI * 2)
            ctx.fillStyle = g
            ctx.fill()
        }
    }

    // ── Audio ring ───────────────────────────────────────────────── //
    Rectangle {
        anchors.centerIn: parent
        width: root.bodyRadius * 2 * (1.10 + root.orbAmplitude * 0.42)
        height: width
        radius: width / 2
        color: "transparent"
        border.color: Qt.rgba(1, 1, 1, 0.35)
        border.width: 1.5
        opacity: root.orbAmplitude * 0.7
        visible: root.orbAmplitude > 0.04
        antialiasing: true
    }

    // Mouse handling lives in OrbWindow (see nixorb/ui/orb_window.py):
    // drag to move, double-click to activate, right-click for the menu,
    // scroll to change opacity. A MouseArea here would swallow all of it.

    // No caption: at this size it collided with the audio ring and was
    // illegible on a light desktop. The glow's colour and motion carry
    // the state, and the tray tooltip spells it out (see
    // nixorb/ui/tray_icon.py).
}
