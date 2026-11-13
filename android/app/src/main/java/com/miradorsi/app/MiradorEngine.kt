package com.miradorsi.app

import android.content.Context
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject

/** Ce que Mirador SI dit de lui-même au démarrage. */
data class EngineInfo(
    val name: String,
    val version: String,
    val tagline: String,
    val description: String,
    val model: String,
    val hasModel: Boolean,
    val principles: List<String>,
)

data class Source(val id: Int, val title: String, val url: String, val domain: String, val reliability: String)

data class Reply(val text: String, val sources: List<Source>)

/** Mode de la zone de saisie : analyse sur internet, ou discussion avec le modèle embarqué. */
enum class Mode(val key: String) { ANALYSE("analyse"), LOCAL("local") }

/**
 * Fait tourner Mirador SI (le code Python du projet, embarqué grâce à Chaquopy) sur le téléphone.
 * Une seule requête à la fois : le modèle et les outils web ne sont pas partagés entre fils.
 */
class MiradorEngine(context: Context) {
    private val appContext = context.applicationContext
    private val mutex = Mutex()
    private var bridge: PyObject? = null

    suspend fun start(): EngineInfo = withContext(Dispatchers.IO) {
        mutex.withLock {
            if (!Python.isStarted()) Python.start(AndroidPlatform(appContext))
            val module = bridge ?: Python.getInstance().getModule("mirador_app.bridge").also { bridge = it }
            val json = JSONObject(module.callAttr("start", appContext.filesDir.absolutePath).toString())
            EngineInfo(
                name = json.optString("name", "Mirador SI"),
                version = json.optString("version"),
                tagline = json.optString("tagline"),
                description = json.optString("description"),
                model = json.optString("model"),
                hasModel = json.optBoolean("has_model"),
                principles = json.optJSONArray("principles").toStringList(),
            )
        }
    }

    suspend fun ask(message: String, mode: Mode, depth: Int, history: List<Pair<String, String>>): Reply =
        withContext(Dispatchers.IO) {
            mutex.withLock {
                val module = bridge ?: return@withLock Reply("Mirador SI n'a pas pu démarrer.", emptyList())
                val past = JSONArray()
                history.forEach { (question, answer) -> past.put(JSONArray().put(question).put(answer)) }
                val json = JSONObject(module.callAttr("ask", message, mode.key, depth, past.toString()).toString())
                val sources = mutableListOf<Source>()
                val array = json.optJSONArray("sources") ?: JSONArray()
                for (i in 0 until array.length()) {
                    val s = array.getJSONObject(i)
                    sources += Source(s.optInt("id"), s.optString("title"), s.optString("url"),
                        s.optString("domain"), s.optString("reliability"))
                }
                Reply(json.optString("text"), sources)
            }
        }
}

private fun JSONArray?.toStringList(): List<String> =
    if (this == null) emptyList() else (0 until length()).map { getString(it) }
