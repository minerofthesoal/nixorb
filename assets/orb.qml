// assets/orb.qml
// Audio-reactive particle orb with GLSL shader glow effect.
// Requires Qt 6.5+ with QtQuick.Effects or ShaderEffect.

import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Particles 2.15

Item {
    id: root
    width: 120
    height: 120
    layer.enabled: true

    // ---------------------------------------------------------------- #
    //  Properties from OrbBridge                                        #
    // ---------------------------------------------------------------- #
    property real amplitude:  orbBridge.amplitude   // 0.0–1.0
    property string orbColor: orbBridge.color       // hex string
    property string orbState: orbBridge.state

    // ---------------------------------------------------------------- #
    //  Core glow orb via ShaderEffect                                   #
    // ---------------------------------------------------------------- #
    ShaderEffect {
        id: glowShader
        anchors.fill: parent
        blending: true

        property real time:      0.0
        property real amplitude: root.amplitude
        property color baseColor: Qt.color(root.orbColor)

        NumberAnimation on time {
            from: 0; to: 6.2832
            duration: 3000
            loops: Animation.Infinite
            running: true
        }

        // Vertex shader — standard passthrough
        vertexShader: "shaders/orb_glow.vert.qsb"

        // Fragment shader — animated voronoi glow
        fragmentShader: "shaders/orb_glow.frag.qsb"
    }

    // ---------------------------------------------------------------- #
    //  Particle system for audio spikes                                 #
    // ---------------------------------------------------------------- #
    ParticleSystem {
        id: particleSystem
        running: root.orbState === "speaking"
    }

    ImageParticle {
        system: particleSystem
        source: "qrc:///qtquick/particleresources/glowdot.png"
        color: root.orbColor
        colorVariation: 0.3
        alpha: 0.8
        alphaVariation: 0.2
        sizeTable: "qrc:///qtquick/particleresources/sine.png"
    }

    Emitter {
        id: particleEmitter
        system: particleSystem
        anchors.centerIn: parent
        emitRate: Math.floor(root.amplitude * 80)
        lifeSpan: 800
        lifeSpanVariation: 400

        size: 8
        sizeVariation: 6
        endSize: 1

        velocity: AngleDirection {
            angleVariation: 360
            magnitude: 40 + root.amplitude * 60
            magnitudeVariation: 20
        }
    }

    // ---------------------------------------------------------------- #
    //  Pulsing ring                                                     #
    // ---------------------------------------------------------------- #
    Rectangle {
        id: pulseRing
        anchors.centerIn: parent
        width:  80 + root.amplitude * 30
        height: width
        radius: width / 2
        color:  "transparent"
        border.color: root.orbColor
        border.width: 2
        opacity: 0.4 + root.amplitude * 0.4

        Behavior on width     { SmoothedAnimation { velocity: 100 } }
        Behavior on opacity   { SmoothedAnimation { velocity: 2  } }
        Behavior on border.color { ColorAnimation { duration: 300 } }
    }

    // ---------------------------------------------------------------- #
    //  Mouse interaction                                                #
    // ---------------------------------------------------------------- #
    MouseArea {
        anchors.fill: parent
        acceptedButtons: Qt.LeftButton | Qt.RightButton

        onClicked: function(mouse) {
            if (mouse.button === Qt.LeftButton) {
                orbBridge.clicked()
            }
        }
        onDoubleClicked: {
            orbBridge.clicked()
        }
        // Right-click context menu
        onPressAndHold: contextMenu.open()
    }

    Menu {
        id: contextMenu
        MenuItem {
            text: "⚙ Settings"
            onTriggered: orbBridge.openSettings()
        }
        MenuItem {
            text: "🔇 Mute Mic"
            checkable: true
        }
        MenuSeparator {}
        MenuItem {
            text: "✕ Quit NixOrb"
            onTriggered: Qt.quit()
        }
    }

    // ---------------------------------------------------------------- #
    //  State-based animations                                           #
    // ---------------------------------------------------------------- #
    states: [
        State {
            name: "listening"
            when: orbState === "listening"
            PropertyChanges { target: pulseRing; border.width: 3 }
        },
        State {
            name: "speaking"
            when: orbState === "speaking"
        }
    ]

    transitions: Transition {
        ColorAnimation { duration: 400 }
        NumberAnimation { properties: "border.width"; duration: 200 }
    }
}
