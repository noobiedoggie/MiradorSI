package com.miradorsi.app

import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.graphics.vector.addPathNodes
import androidx.compose.ui.unit.dp

/** Icônes dessinées pour Mirador SI (tracés 24×24 de style Material, logo original). */
object MiradorIcons {
    private fun filled(name: String, path: String): ImageVector =
        ImageVector.Builder(name, 24.dp, 24.dp, 24f, 24f)
            .addPath(addPathNodes(path), fill = SolidColor(Color.Black))
            .build()

    val Mic = filled(
        "Mic",
        "M12,14c1.66,0 2.99,-1.34 2.99,-3L15,5c0,-1.66 -1.34,-3 -3,-3S9,3.34 9,5v6c0,1.66 1.34,3 3,3z" +
            "M17.3,11c0,3 -2.54,5.1 -5.3,5.1S6.7,14 6.7,11H5c0,3.41 2.72,6.23 6,6.72V21h2v-3.28" +
            "c3.28,-0.48 6,-3.3 6,-6.72h-1.7z",
    )

    val Laptop = filled(
        "Laptop",
        "M20,18c1.1,0 1.99,-0.9 1.99,-2L22,6c0,-1.1 -0.9,-2 -2,-2H4c-1.1,0 -2,0.9 -2,2v10" +
            "c0,1.1 0.9,2 2,2H0v2h24v-2h-4zM4,6h16v10H4V6z",
    )

    val Waveform = filled(
        "Waveform",
        "M7,18h2V6H7v12zM11,22h2V2h-2v20zM3,14h2v-4H3v4zM15,18h2V6h-2v12zM19,10v4h2v-4h-2z",
    )

    val ArrowUp = filled(
        "ArrowUp",
        "M4,12l1.41,1.41L11,7.83V20h2V7.83l5.58,5.59L20,12l-8,-8 -8,8z",
    )

    val Library = filled(
        "Library",
        "M4,6H2v14c0,1.1 0.9,2 2,2h14v-2H4V6zM20,2H8c-1.1,0 -2,0.9 -2,2v12c0,1.1 0.9,2 2,2h12" +
            "c1.1,0 2,-0.9 2,-2V4c0,-1.1 -0.9,-2 -2,-2zM19,11H9V9h10v2zM15,15H9v-2h6v2zM19,7H9V5h10v2z",
    )

    val NewChat = filled(
        "NewChat",
        "M3,17.25V21h3.75L17.81,9.94l-3.75,-3.75L3,17.25zM20.71,7.04c0.39,-0.39 0.39,-1.02 0,-1.41" +
            "l-2.34,-2.34c-0.39,-0.39 -1.02,-0.39 -1.41,0l-1.83,1.83 3.75,3.75 1.83,-1.83z",
    )

    val Download = filled("Download", "M19,9h-4V3H9v6H5l7,7 7,-7zM5,18v2h14v-2H5z")

    /** Le logo : une tour de guet, deux fenêtres allumées. */
    fun tower(frame: Color, light: Color): ImageVector =
        ImageVector.Builder("Mirador", 48.dp, 48.dp, 48f, 48f)
            .addPath(addPathNodes("M12,15 L24,6 L36,15 Z"), fill = SolidColor(frame))
            .addPath(
                addPathNodes("M14,15 h20 v10 h-20 Z"),
                stroke = SolidColor(frame), strokeLineWidth = 2.6f, strokeLineJoin = StrokeJoin.Round,
            )
            .addPath(addPathNodes("M18.5,18 h4 v4.5 h-4 Z M25.5,18 h4 v4.5 h-4 Z"), fill = SolidColor(light))
            .addPath(
                addPathNodes("M11,25 h26 M17,25 L13,44 M31,25 L35,44 M16,31 L32,38 M32,31 L16,38 M9,44 h30"),
                stroke = SolidColor(frame), strokeLineWidth = 2.6f, strokeLineCap = StrokeCap.Round,
            )
            .build()
}
