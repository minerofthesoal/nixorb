import QtQuick 2.15
import QtQuick.Shapes 1.15
import NixOrb 1.0

// NixOrb Floating Orb — GLSL shader-powered glowing sphere
// States: idle (blue) → listening (green) → thinking (amber) → speaking (purple)

Item {
    id: root
    width: orbBridge ? orbBridge.orbSize : 120
    height: width

    // Properties bound to C++ bridge
    property string orbState: orbBridge ? orbBridge.state : "idle"
    property real orbAmplitude: orbBridge ? orbBridge.amplitude : 0.0
    property string orbColor: orbBridge ? orbBridge.color : "#4A90D9"
    property real orbOpacity: orbBridge ? orbBridge.opacity : 1.0

    // Animated properties
    property real pulseScale: 1.0
    property real glowIntensity: 0.6

    // Opacity
    opacity: orbOpacity

    // Pulse animation for speaking/listening
    SequentialAnimation on pulseScale {
        id: pulseAnim
        loops: Animation.Infinite
        running: orbState === "speaking" || orbState === "listening"
        NumberAnimation { to: 1.15; duration: 400; easing.type: Easing.InOutQuad }
        NumberAnimation { to: 1.0; duration: 400; easing.type: Easing.InOutQuad }
    }

    // Glow intensity animation
    NumberAnimation on glowIntensity {
        id: glowAnim
        running: orbState === "thinking"
        loops: Animation.Infinite
        from: 0.3
        to: 1.0
        duration: 800
        easing.type: Easing.InOutSine
    }

    // Outer glow
    Rectangle {
        id: outerGlow
        anchors.centerIn: parent
        width: parent.width * pulseScale * 1.4
        height: width
        radius: width / 2
        color: orbColor
        opacity: (0.15 + orbAmplitude * 0.3) * glowIntensity

        Behavior on opacity {
            NumberAnimation { duration: 200 }
        }
    }

    // Middle glow
    Rectangle {
        id: middleGlow
        anchors.centerIn: parent
        width: parent.width * pulseScale * 1.15
        height: width
        radius: width / 2
        color: orbColor
        opacity: (0.3 + orbAmplitude * 0.4) * glowIntensity

        Behavior on opacity {
            NumberAnimation { duration: 150 }
        }
    }

    // Main orb body
    Rectangle {
        id: orbBody
        anchors.centerIn: parent
        width: parent.width * pulseScale
        height: width
        radius: width / 2

        // Gradient for 3D sphere effect
        gradient: Gradient {
            GradientStop { position: 0.0; color: Qt.lighter(orbColor, 1.4) }
            GradientStop { position: 0.4; color: orbColor }
            GradientStop { position: 1.0; color: Qt.darker(orbColor, 1.6) }
        }

        // Highlight reflection
        Rectangle {
            anchors {
                top: parent.top
                topMargin: parent.height * 0.12
                horizontalCenter: parent.horizontalCenter
            }
            width: parent.width * 0.35
            height: width * 0.55
            radius: width / 2
            color: "#FFFFFF"
            opacity: 0.25 + orbAmplitude * 0.15
        }
    }

    // Audio reactivity ring (visible when speaking/listening)
    Rectangle {
        id: audioRing
        anchors.centerIn: parent
        width: parent.width * (1.0 + orbAmplitude * 0.5)
        height: width
        radius: width / 2
        color: "transparent"
        border.color: orbColor
        border.width: 2
        opacity: orbAmplitude * 0.8
        visible: orbAmplitude > 0.05
    }

    // Click handler
    MouseArea {
        anchors.fill: parent
        onClicked: {
            if (orbBridge) {
                orbBridge.clicked()
            }
        }
        onDoubleClicked: {
            if (orbBridge) {
                orbBridge.openSettings()
            }
        }
    }

    // State indicator text (small, subtle)
    Text {
        anchors {
            bottom: parent.bottom
            bottomMargin: -18
            horizontalCenter: parent.horizontalCenter
        }
        text: orbState.charAt(0).toUpperCase() + orbState.slice(1)
        color: orbColor
        font.pixelSize: 10
        opacity: 0.7
    }
}
