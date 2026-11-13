package com.miradorsi.app

import android.content.Context
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL

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

/**
 * Réponse de Mirador SI. Si un modèle ouvert est choisi, `invite` est la conversation qu'il doit
 * rédiger (sources numérotées + question) ; `text` reste la réponse de secours.
 * `genre` : « chat » (discussion), « sources » (passages trouvés) ou « free » (aucune source).
 */
data class Reply(
    val text: String,
    val sources: List<Source>,
    val invite: List<Pair<String, String>>? = null,
    val genre: String = "",
    val sourcesInvite: List<Source> = emptyList(),
)

/** La bibliothèque hors ligne : articles de Wikipédia sur le téléphone, pour répondre sans internet. */
data class LibraryState(val installed: Boolean, val description: String)

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

    suspend fun libraryState(): LibraryState = withContext(Dispatchers.IO) {
        mutex.withLock {
            val module = bridge ?: return@withLock LibraryState(false, "")
            JSONObject(module.callAttr("library_info").toString()).toLibraryState()
        }
    }

    /** Branche un fichier de bibliothèque déjà présent sur le téléphone. */
    suspend fun useLibrary(file: File): LibraryState = withContext(Dispatchers.IO) {
        mutex.withLock {
            val module = bridge ?: throw IOException("Mirador SI n'a pas démarré")
            JSONObject(module.callAttr("library_load", file.absolutePath).toString()).toLibraryState()
        }
    }

    /**
     * Télécharge la bibliothèque (une centaine de Mo), la vérifie puis la branche. Le téléchargement
     * se fait hors du verrou : on peut continuer à utiliser Mirador pendant ce temps.
     */
    suspend fun downloadLibrary(onProgress: (Long, Long) -> Unit): LibraryState = withContext(Dispatchers.IO) {
        val dir = File(appContext.filesDir, "bibliotheque").apply { mkdirs() }
        val part = File(dir, "bibliotheque-fr.db.part")
        val target = File(dir, "bibliotheque-fr.db")
        val connection = (URL(URL_BIBLIOTHEQUE).openConnection() as HttpURLConnection).apply {
            connectTimeout = 20_000
            readTimeout = 60_000
            instanceFollowRedirects = true
            setRequestProperty("User-Agent", "MiradorSI-Android")
        }
        try {
            if (connection.responseCode != HttpURLConnection.HTTP_OK) {
                throw IOException("le serveur a répondu ${connection.responseCode}")
            }
            val total = connection.contentLengthLong
            var done = 0L
            var shown = -1L
            connection.inputStream.use { input ->
                part.outputStream().use { output ->
                    val buffer = ByteArray(64 * 1024)
                    while (true) {
                        val n = input.read(buffer)
                        if (n < 0) break
                        output.write(buffer, 0, n)
                        done += n
                        val percent = if (total > 0) done * 100 / total else done / 1_000_000
                        if (percent != shown) { shown = percent; onProgress(done, total) }
                    }
                }
            }
        } finally {
            connection.disconnect()
        }
        if (target.exists()) target.delete()
        if (!part.renameTo(target)) throw IOException("impossible d'enregistrer la bibliothèque")
        try {
            useLibrary(target)
        } catch (e: Throwable) {
            target.delete()
            throw IOException("fichier de bibliothèque invalide", e)
        }
    }

    private fun sources(array: JSONArray?): List<Source> = (0 until (array?.length() ?: 0)).map {
        val s = array!!.getJSONObject(it)
        Source(s.optInt("id"), s.optString("title"), s.optString("url"), s.optString("domain"), s.optString("reliability"))
    }

    /** Le téléphone a-t-il internet ? Le mode de base ne cherche en ligne que dans ce cas. */
    private fun enLigne(): Boolean = try {
        val reseau = appContext.getSystemService(ConnectivityManager::class.java)
        val capacites = reseau?.getNetworkCapabilities(reseau.activeNetwork)
        capacites?.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET) == true
    } catch (e: SecurityException) {
        true
    }

    suspend fun ask(message: String, mode: Mode, depth: Int, history: List<Pair<String, String>>,
                    llm: Boolean = false): Reply =
        withContext(Dispatchers.IO) {
            mutex.withLock {
                val module = bridge ?: return@withLock Reply("Mirador SI n'a pas pu démarrer.", emptyList())
                val past = JSONArray()
                history.forEach { (question, answer) -> past.put(JSONArray().put(question).put(answer)) }
                val json = JSONObject(
                    module.callAttr("ask", message, mode.key, depth, past.toString(), enLigne(), llm).toString())
                val invite = json.optJSONArray("llm")?.let { a ->
                    (0 until a.length()).map { a.getJSONObject(it).let { m -> m.optString("role") to m.optString("content") } }
                }
                Reply(json.optString("text"), sources(json.optJSONArray("sources")), invite,
                    json.optString("llm_kind"), sources(json.optJSONArray("llm_sources")))
            }
        }
}

/** Où l'application télécharge la bibliothèque (construite par la tâche GitHub « Bibliothèque hors ligne »). */
const val URL_BIBLIOTHEQUE = "https://github.com/noobiedoggie/MiradorSI/releases/download/bibliotheque/bibliotheque-fr.db"

private fun JSONObject.toLibraryState() =
    LibraryState(optBoolean("installed"), optString("description"))

private fun JSONArray?.toStringList(): List<String> =
    if (this == null) emptyList() else (0 until length()).map { getString(it) }
