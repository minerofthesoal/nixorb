import QtQuick 2.15

// NixOrb floating orb.
//
// The sphere is drawn once per colour change into a Canvas: a real radial
// gradient with an off-axis key light, limb darkening and a bounce light
// reads as a sphere in a way stacked Rectangles never did. Everything that
// moves every frame (bloom, ring, pulse) is a plain Item transform, so the
// Canvas is not repainted at 60fps.
//
// Palette lives in nixorb/ui/orb_window.py (STATE_COLORS).

Item {
    id: root
    width: orbBridge ? orbBridge.orbSize : 88
    height: width

    property string orbState: orbBridge ? orbBridge.state : "idle"
    property real orbAmplitude: orbBridge ? orbBridge.amplitude : 0.0
    property color orbColor: orbBridge ? orbBridge.color : "#C96442"
    property real orbOpacity: orbBridge ? orbBridge.opacity : 1.0

    // Sphere occupies the middle; the rest of the window is bloom headroom.
    readonly property real bodyRadius: width * 0.30
    readonly property real cx: width / 2
    readonly property real cy: height / 2

    property real pulse: 1.0
    property real glow: 0.55

    opacity: orbOpacity
    antialiasing: true

    // ── Colour helpers ───────────────────────────────────────────── //
    // Mixing toward white/black keeps saturated clays from clipping the way
    // Qt.lighter() does.
    function shade(c, amount) {
        var t = Math.abs(amount)
        var to = amount > 0 ? 1.0 : 0.0
        return Qt.rgba(c.r + (to - c.r) * t,
                       c.g + (to - c.g) * t,
                       c.b + (to - c.b) * t, 1.0)
    }

    function css(c, a) {
        return "rgba(" + Math.round(c.r * 255) + ","
                       + Math.round(c.g * 255) + ","
                       + Math.round(c.b * 255) + "," + a + ")"
    }

    // ── Animation ────────────────────────────────────────────────── //
    SequentialAnimation on pulse {
        loops: Animation.Infinite
        running: orbState === "speaking" || orbState === "listening"
        NumberAnimation { to: 1.045; duration: 520; easing.type: Easing.InOutSine }
        NumberAnimation { to: 1.000; duration: 520; easing.type: Easing.InOutSine }
    }

    SequentialAnimation on glow {
        loops: Animation.Infinite
        running: orbState === "thinking"
        NumberAnimation { to: 1.00; duration: 700; easing.type: Easing.InOutSine }
        NumberAnimation { to: 0.35; duration: 700; easing.type: Easing.InOutSine }
    }

    // Slow drift so the orb never looks frozen while idle.
    SequentialAnimation on glow {
        loops: Animation.Infinite
        running: orbState === "idle"
        NumberAnimation { to: 0.62; duration: 2600; easing.type: Easing.InOutSine }
        NumberAnimation { to: 0.44; duration: 2600; easing.type: Easing.InOutSine }
    }

    Behavior on orbColor { ColorAnimation { duration: 260 } }

    // ── Outer bloom ──────────────────────────────────────────────── //
    Rectangle {
        anchors.centerIn: parent
        width: root.bodyRadius * 2 * 1.62 * root.pulse
        height: width
        radius: width / 2
        color: root.orbColor
        opacity: (0.055 + root.orbAmplitude * 0.10) * root.glow
        antialiasing: true
    }

    Rectangle {
        anchors.centerIn: parent
        width: root.bodyRadius * 2 * 1.30 * root.pulse
        height: width
        radius: width / 2
        color: root.orbColor
        opacity: (0.11 + root.orbAmplitude * 0.16) * root.glow
        antialiasing: true
        Behavior on opacity { NumberAnimation { duration: 140 } }
    }

    // ── Sphere ───────────────────────────────────────────────────── //
    Canvas {
        id: sphere
        anchors.fill: parent
        antialiasing: true
        smooth: true
        renderStrategy: Canvas.Cooperative
        scale: root.pulse

        Connections {
            target: root
            function onOrbColorChanged() { sphere.requestPaint() }
        }

        onPaint: {
            var ctx = getContext("2d")
            ctx.reset()

            var c = root.orbColor
            var r = root.bodyRadius
            var cx = root.cx
            var cy = root.cy

            // Key light sits up and to the left, as it does on every sphere
            // anybody has ever found convincing.
            var lx = cx - r * 0.38
            var ly = cy - r * 0.42

            var body = ctx.createRadialGradient(lx, ly, r * 0.04, cx, cy, r * 1.12)
            body.addColorStop(0.00, root.css(root.shade(c, 0.52), 1.0))
            body.addColorStop(0.22, root.css(root.shade(c, 0.20), 1.0))
            body.addColorStop(0.52, root.css(c, 1.0))
            body.addColorStop(0.82, root.css(root.shade(c, -0.34), 1.0))
            body.addColorStop(1.00, root.css(root.shade(c, -0.55), 1.0))

            ctx.beginPath()
            ctx.arc(cx, cy, r, 0, Math.PI * 2)
            ctx.fillStyle = body
            ctx.fill()

            // Everything below is confined to the sphere.
            ctx.save()
            ctx.beginPath()
            ctx.arc(cx, cy, r, 0, Math.PI * 2)
            ctx.clip()

            // Bounce light along the lower-right limb — the cue that stops a
            // shaded circle from reading as a flat disc.
            var bx = cx + r * 0.42
            var by = cy + r * 0.52
            var bounce = ctx.createRadialGradient(bx, by, r * 0.05, bx, by, r * 0.95)
            bounce.addColorStop(0.00, root.css(root.shade(c, 0.42), 0.55))
            bounce.addColorStop(0.55, root.css(root.shade(c, 0.20), 0.16))
            bounce.addColorStop(1.00, root.css(c, 0.0))
            ctx.fillStyle = bounce
            ctx.fillRect(0, 0, width, height)

            // Specular highlight: an ellipse, tilted, soft-edged.
            ctx.save()
            ctx.translate(cx - r * 0.30, cy - r * 0.40)
            ctx.rotate(-0.42)
            ctx.scale(1.0, 0.62)
            var spec = ctx.createRadialGradient(0, 0, 0, 0, 0, r * 0.52)
            spec.addColorStop(0.00, "rgba(255,255,255,0.62)")
            spec.addColorStop(0.42, "rgba(255,255,255,0.16)")
            spec.addColorStop(1.00, "rgba(255,255,255,0.0)")
            ctx.fillStyle = spec
            ctx.beginPath()
            ctx.arc(0, 0, r * 0.52, 0, Math.PI * 2)
            ctx.fill()
            ctx.restore()

            ctx.restore()

            // Crisp terminator at the edge, so the orb keeps its silhouette
            // against a light desktop as well as a dark one.
            ctx.beginPath()
            ctx.arc(cx, cy, r - 0.5, 0, Math.PI * 2)
            ctx.lineWidth = 1
            ctx.strokeStyle = root.css(root.shade(c, -0.42), 0.7)
            ctx.stroke()
        }
    }

    // Voice level brightens the sphere from within. White, so it never needs
    // repainting when the state colour changes — only its opacity animates.
    Canvas {
        id: sheen
        anchors.fill: parent
        antialiasing: true
        smooth: true
        renderStrategy: Canvas.Cooperative
        scale: root.pulse
        opacity: root.orbAmplitude * 0.42
        visible: opacity > 0.01
        Behavior on opacity { NumberAnimation { duration: 90 } }

        onPaint: {
            var ctx = getContext("2d")
            ctx.reset()
            var r = root.bodyRadius
            var hx = root.cx - r * 0.18
            var hy = root.cy - r * 0.22
            var g = ctx.createRadialGradient(hx, hy, 0, hx, hy, r * 0.98)
            g.addColorStop(0.00, "rgba(255,255,255,0.85)")
            g.addColorStop(0.45, "rgba(255,255,255,0.30)")
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
        border.color: root.shade(root.orbColor, 0.28)
        border.width: 1.5
        opacity: root.orbAmplitude * 0.7
        visible: root.orbAmplitude > 0.04
        antialiasing: true
    }

    // Mouse handling lives in OrbWindow (see nixorb/ui/orb_window.py):
    // drag to move, double-click to activate, right-click for the menu,
    // scroll to change opacity. A MouseArea here would swallow all of it.

    // No caption: at this size it collided with the audio ring and was
    // illegible on a light desktop. The colour carries the state, and the
    // tray tooltip spells it out (see nixorb/ui/tray_icon.py).
}
