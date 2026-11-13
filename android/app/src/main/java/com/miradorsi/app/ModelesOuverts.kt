package com.miradorsi.app

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL

/**
 * Modèles de langage ouverts que Mirador SI peut faire tourner sur le téléphone (format GGUF,
 * moteur llama.cpp). Ils sont téléchargés depuis Hugging Face à la demande, jamais inclus dans l'APK.
 * Licences libres : le fichier vient directement de l'éditeur ou de son dépôt de conversion.
 */
data class ModeleOuvert(
    val id: String,
    val nom: String,
    val editeur: String,
    val role: String,
    val resume: String,
    val depot: String,
    val fichier: String,
    val octets: Long,
    val licence: String,
    val contexte: Int,
) {
    val url: String get() = "https://huggingface.co/$depot/resolve/main/$fichier"
    val taille: String get() = if (octets >= 1_000_000_000) "%.1f Go".format(octets / 1e9) else "${octets / 1_000_000} Mo"
}

object Catalogue {
    val MODELES = listOf(
        ModeleOuvert(
            id = "gemma3-1b", nom = "Gemma 3 1B", editeur = "Google", role = "Rapide",
            resume = "Le plus léger : répond vite, même sur un téléphone modeste.",
            depot = "unsloth/gemma-3-1b-it-GGUF", fichier = "gemma-3-1b-it-Q4_K_M.gguf", octets = 806_058_272,
            licence = "Gemma Terms of Use", contexte = 4096,
        ),
        ModeleOuvert(
            id = "qwen35-2b", nom = "Qwen3.5 2B", editeur = "Alibaba", role = "Équilibré",
            resume = "Bon compromis entre vitesse et qualité des réponses.",
            depot = "unsloth/Qwen3.5-2B-GGUF", fichier = "Qwen3.5-2B-Q4_K_M.gguf", octets = 1_280_835_840,
            licence = "Apache 2.0", contexte = 4096,
        ),
        ModeleOuvert(
            id = "ministral3-3b", nom = "Ministral 3 3B", editeur = "Mistral AI (France)", role = "Puissant",
            resume = "Le plus fort en français ; plus lent, pour les téléphones récents.",
            depot = "mistralai/Ministral-3-3B-Instruct-2512-GGUF", fichier = "Ministral-3-3B-Instruct-2512-Q4_K_M.gguf",
            octets = 2_147_023_008, licence = "Apache 2.0", contexte = 3072,
        ),
    )

    fun parId(id: String?): ModeleOuvert? = MODELES.firstOrNull { it.id == id }
}

/** Appels vers le moteur natif (cpp/jni_moteur.cpp). Les textes circulent en octets UTF-8. */
object MoteurNatif {
    @Volatile private var pret: Boolean? = null

    /** Raison de l'échec du chargement de la bibliothèque native, s'il y en a eu un. */
    @Volatile var erreurChargement: String = ""
        private set

    /** Charge la bibliothèque native une seule fois ; false si elle manque (appareil non pris en charge). */
    fun disponible(context: Context): Boolean {
        pret?.let { return it }
        synchronized(this) {
            pret?.let { return it }
            val ok = try {
                System.loadLibrary("mirador_llm")
                initialiser(context.applicationInfo.nativeLibraryDir)
                true
            } catch (e: Throwable) {
                erreurChargement = e.toString()
                android.util.Log.e("MiradorSI", "moteur natif indisponible", e)
                false
            }
            pret = ok
            return ok
        }
    }

    external fun initialiser(dossier: String)
    external fun ouvrir(chemin: String, nCtx: Int, nThreads: Int): Long
    external fun derniereErreur(): String
    external fun description(handle: Long): String
    external fun formater(handle: Long, roles: Array<String>, contenus: Array<ByteArray>): ByteArray
    external fun generer(handle: Long, prompt: ByteArray, maxTokens: Int, temperature: Float, rappel: Rappel): ByteArray
    external fun fermer(handle: Long)
}

/** Reçoit le texte au fur et à mesure ; renvoie false pour arrêter la génération. */
fun interface Rappel {
    fun morceau(octets: ByteArray): Boolean
}

sealed class EtatModele {
    data object Absent : EtatModele()
    data class Telechargement(val fait: Long, val total: Long) : EtatModele()
    data object Installe : EtatModele()
    data class Erreur(val message: String) : EtatModele()
}

/** Téléchargement, choix et exécution des modèles ouverts. Un seul modèle chargé en mémoire à la fois. */
class GestionnaireModeles(context: Context) {
    private val appContext = context.applicationContext
    private val dossier = File(appContext.filesDir, "modeles").apply { mkdirs() }
    private val preferences = appContext.getSharedPreferences("modeles", Context.MODE_PRIVATE)
    private val verrou = Mutex()
    private var charge: Pair<String, Long>? = null  // (id du modèle, handle natif)
    @Volatile private var arreter = false

    fun fichier(m: ModeleOuvert) = File(dossier, m.fichier)
    fun installe(m: ModeleOuvert) = fichier(m).length() == m.octets

    /** Le modèle choisi pour répondre, ou null pour Vigie (le modèle maison, sans téléchargement). */
    var actif: ModeleOuvert?
        get() = Catalogue.parId(preferences.getString("actif", null))?.takeIf { installe(it) }
        set(value) { preferences.edit().putString("actif", value?.id).apply() }

    fun moteurDisponible(): Boolean = MoteurNatif.disponible(appContext)

    suspend fun telecharger(m: ModeleOuvert, progression: (Long, Long) -> Unit): Unit = withContext(Dispatchers.IO) {
        val partiel = File(dossier, m.fichier + ".part")
        val libre = dossier.usableSpace + partiel.length()
        if (libre < m.octets + 200_000_000) {
            throw IOException("pas assez de place (il faut ${m.taille} libres)")
        }
        var tentative = 0
        while (true) {
            try {
                telechargerUneFois(m, partiel, progression)
                break
            } catch (e: IOException) {
                if (++tentative >= 4) throw e
                Thread.sleep(2000L * tentative)  // coupure réseau : on reprend là où on s'était arrêté
            }
        }
        if (partiel.length() != m.octets) {
            partiel.delete()
            throw IOException("fichier incomplet, réessaie")
        }
        if (!partiel.renameTo(fichier(m))) throw IOException("enregistrement impossible")
    }

    private fun telechargerUneFois(m: ModeleOuvert, partiel: File, progression: (Long, Long) -> Unit) {
        var deja = partiel.length()
        if (deja >= m.octets) return
        val connexion = URL(m.url).openConnection() as HttpURLConnection
        connexion.instanceFollowRedirects = true
        connexion.connectTimeout = 20_000
        connexion.readTimeout = 60_000
        connexion.setRequestProperty("User-Agent", "MiradorSI-Android")
        if (deja > 0) connexion.setRequestProperty("Range", "bytes=$deja-")
        try {
            val code = connexion.responseCode
            if (code == 200) deja = 0  // le serveur ignore la reprise : on recommence
            else if (code != 206) throw IOException("HTTP $code")
            connexion.inputStream.use { entree ->
                FileOutputStream(partiel, deja > 0).use { sortie ->
                    val tampon = ByteArray(256 * 1024)
                    var fait = deja
                    var dernier = 0L
                    while (true) {
                        val n = entree.read(tampon)
                        if (n < 0) break
                        sortie.write(tampon, 0, n)
                        fait += n
                        if (fait - dernier >= 4_000_000) {
                            dernier = fait
                            progression(fait, m.octets)
                        }
                    }
                    progression(fait, m.octets)
                }
            }
        } finally {
            connexion.disconnect()
        }
    }

    suspend fun supprimer(m: ModeleOuvert) {
        verrou.withLock {
            if (charge?.first == m.id) liberer()
            fichier(m).delete()
            File(dossier, m.fichier + ".part").delete()
            if (preferences.getString("actif", null) == m.id) actif = null
        }
    }

    /** Libère la mémoire du modèle chargé (appelé quand on revient à Vigie ou qu'on change de modèle). */
    private fun liberer() {
        charge?.let { MoteurNatif.fermer(it.second) }
        charge = null
    }

    suspend fun decharger() {
        verrou.withLock { liberer() }
    }

    private fun handle(m: ModeleOuvert): Long {
        charge?.let { if (it.first == m.id) return it.second }
        liberer()
        val coeurs = Runtime.getRuntime().availableProcessors()
        val fils = if (coeurs >= 8) 4 else (coeurs - 1).coerceIn(1, 4)
        val h = MoteurNatif.ouvrir(fichier(m).absolutePath, m.contexte, fils)
        if (h == 0L) throw IOException(MoteurNatif.derniereErreur().ifBlank { "modèle illisible" })
        charge = m.id to h
        return h
    }

    /** Prépare le modèle (chargement en mémoire), pour que la première réponse soit plus rapide. */
    suspend fun prechauffer(m: ModeleOuvert) {
        withContext(Dispatchers.Default) {
            if (moteurDisponible()) {
                verrou.withLock { runCatching { handle(m) } }
            }
        }
    }

    fun interrompre() { arreter = true }

    /**
     * Rédige une réponse avec le modèle `m` à partir d'une conversation (rôle, texte).
     * `morceau` reçoit le texte déjà écrit, au fur et à mesure.
     */
    suspend fun rediger(m: ModeleOuvert, messages: List<Pair<String, String>>, maxTokens: Int = 600,
                        morceau: (String) -> Unit): String = withContext(Dispatchers.Default) {
        if (!moteurDisponible()) throw IOException("moteur natif indisponible sur cet appareil")
        verrou.withLock {
            val h = handle(m)
            arreter = false
            val prompt = MoteurNatif.formater(h, messages.map { it.first }.toTypedArray(),
                messages.map { it.second.toByteArray(Charsets.UTF_8) }.toTypedArray())
            val texte = StringBuilder()
            val sortie = MoteurNatif.generer(h, prompt, maxTokens, 0.3f) { octets ->
                texte.append(String(octets, Charsets.UTF_8))
                morceau(texte.toString())
                !arreter
            }
            String(sortie, Charsets.UTF_8).trim()
        }
    }
}
