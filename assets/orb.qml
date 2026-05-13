// assets/orb.qml
// NixOrb floating orb — Qt 6.5+  QtQuick 2.15  QtQuick.Particles 2.15
import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Particles 2.15

Item {
    id: root
    readonly property real    amplitude : orbBridge.amplitude
    readonly property string  orbColor  : orbBridge.color
    readonly property string  orbState  : orbBridge.state
    width:  120
    height: 120
    layer.enabled: true

    ShaderEffect {
        id: glowShader
        anchors.fill: parent
        property real  time:      0.0
        property real  amplitude: root.amplitude
        property color baseColor: Qt.color(root.orbColor)
        NumberAnimation on time { from:0; to:6.28318; duration:4000; loops:Animation.Infinite; running:true }
        Behavior on baseColor { ColorAnimation { duration: 350 } }
        vertexShader:   Qt.resolvedUrl("shaders/orb_glow.vert.qsb")
        fragmentShader: Qt.resolvedUrl("shaders/orb_glow.frag.qsb")
        blending: true
    }

    Rectangle {
        id: pulseRing
        anchors.centerIn: parent
        width:  76 + root.amplitude * 36; height: width; radius: width/2
        color: "transparent"
        border.color: root.orbColor
        border.width: 1.5 + root.amplitude * 1.5
        opacity: 0.30 + root.amplitude * 0.45
        Behavior on width        { SmoothedAnimation { velocity:120 } }
        Behavior on opacity      { SmoothedAnimation { velocity:3   } }
        Behavior on border.color { ColorAnimation    { duration:350 } }
    }

    Rectangle {
        anchors.centerIn: parent
        width:  92 + root.amplitude * 22; height: width; radius: width/2
        color: "transparent"
        border.color: root.orbColor; border.width: 0.8
        opacity: 0.12 + root.amplitude * 0.18
        Behavior on width        { SmoothedAnimation { velocity:60 } }
        Behavior on border.color { ColorAnimation { duration:350 } }
    }

    ParticleSystem { id: particles; running: root.orbState === "speaking"; paused: root.orbState !== "speaking" }

    ImageParticle {
        system: particles
        source: "qrc:///qtquick/particleresources/glowdot.png"
        color: root.orbColor; colorVariation:0.25; alpha:0.80; alphaVariation:0.20
        sizeTable: "qrc:///qtquick/particleresources/sine.png"
        Behavior on color { ColorAnimation { duration: 350 } }
    }

    Emitter {
        system: particles; anchors.centerIn: parent
        emitRate: Math.max(0, Math.floor(root.amplitude * 90))
        lifeSpan: 700; lifeSpanVariation: 300
        size: 7; sizeVariation: 5; endSize: 1
        velocity: AngleDirection { angleVariation:360; magnitude:38+root.amplitude*65; magnitudeVariation:18 }
    }

    SequentialAnimation {
        loops: Animation.Infinite; running: root.orbState === "listening"
        NumberAnimation { target:pulseRing; property:"opacity"; to:0.80; duration:500; easing.type:Easing.InOutSine }
        NumberAnimation { target:pulseRing; property:"opacity"; to:0.25; duration:500; easing.type:Easing.InOutSine }
    }

    Rectangle {
        visible: root.orbState === "thinking"
        anchors.centerIn: parent; width:100; height:100; radius:50
        color:"transparent"; border.color:root.orbColor; border.width:2; opacity:0.55
        RotationAnimation on rotation { from:0; to:360; duration:1200; loops:Animation.Infinite; running:root.orbState==="thinking" }
        Rectangle { anchors.centerIn:parent; width:parent.width-4; height:parent.height-4; radius:width/2; color:"#1a1a2e" }
    }

    MouseArea {
        id: mouseArea; anchors.fill:parent; hoverEnabled:true
        acceptedButtons: Qt.LeftButton | Qt.RightButton
        onClicked: function(m) { if (m.button===Qt.LeftButton) orbBridge.clicked() }
        onDoubleClicked: orbBridge.clicked()
        onPressAndHold:  ctxMenu.popup()
        onContainsMouseChanged: root.scale = containsMouse ? 1.08 : 1.0
    }
    Behavior on scale { SmoothedAnimation { velocity:6 } }

    Menu {
        id: ctxMenu
        background: Rectangle { color:"#1a1a2e"; border.color:"#2a2a4e"; radius:6 }
        MenuItem { text:"🎙  Activate";  onTriggered: orbBridge.clicked();       contentItem: Text { text:parent.text; color:"#e0e0e0" } }
        MenuItem { text:"⚙  Settings";  onTriggered: orbBridge.openSettings();  contentItem: Text { text:parent.text; color:"#e0e0e0" } }
        MenuSeparator {}
        MenuItem { text:"✕  Quit";      onTriggered: Qt.quit();                 contentItem: Text { text:parent.text; color:"#e74c3c" } }
    }

    Text {
        anchors { horizontalCenter:parent.horizontalCenter; bottom:parent.bottom; bottomMargin:-18 }
        text: root.orbState; color: root.orbColor
        font.pixelSize:10; opacity: mouseArea.containsMouse ? 0.85 : 0.0
        Behavior on opacity { NumberAnimation { duration:200 } }
    }
}
