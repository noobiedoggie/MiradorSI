package com.miradorsi.app

import androidx.test.ext.junit.runners.AndroidJUnit4
import com.chaquo.python.Python
import androidx.test.platform.app.InstrumentationRegistry
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/** Vérifie, sur un vrai Android, que Python, numpy et le modèle embarqué fonctionnent. */
@RunWith(AndroidJUnit4::class)
class MiradorEngineTest {
    private val engine = MiradorEngine(InstrumentationRegistry.getInstrumentation().targetContext)

    @Test
    fun demarreEtChargeLeModele() = runBlocking {
        val info = engine.start()
        assertEquals("Mirador SI", info.name)
        assertTrue("modèle embarqué absent", info.hasModel)
        assertTrue(info.model, info.model.contains("paramètres"))
    }

    @Test
    fun repondHorsLigneSurSonIdentite() = runBlocking {
        engine.start()
        val reponse = engine.ask("Qui es-tu ?", Mode.LOCAL, 1, emptyList())
        assertTrue("réponse : ${reponse.text}", reponse.text.contains("Mirador"))
    }

    @Test
    fun repondHorsLigneAvecLaBibliotheque() = runBlocking {
        engine.start()
        // une mini-bibliothèque construite sur le téléphone même (SQLite + numpy sous Android)
        val fichier = File(InstrumentationRegistry.getInstrumentation().targetContext.filesDir, "essai-bibliotheque.db")
        val module = Python.getInstance().getModule("miradorsi.library")
        val constructeur = module.callAttr("LibraryBuilder", fichier.absolutePath, "fr", "essai")
        constructeur.callAttr("add", "Trou noir", "En astrophysique, un trou noir est un objet céleste si compact " +
            "que rien ne peut s'en échapper, pas même la lumière.", 1e-5)
        constructeur.callAttr("add", "Lyon", "Lyon est une commune française située au confluent du Rhône et de " +
            "la Saône, dans la région Auvergne-Rhône-Alpes.", 1e-6)
        constructeur.callAttr("close")
        val etat = engine.useLibrary(fichier)
        assertTrue(etat.description, etat.installed && etat.description.startsWith("2 articles"))
        val reponse = engine.ask("C'est quoi un trou noir ?", Mode.LOCAL, 1, emptyList())
        assertTrue(reponse.text, reponse.text.startsWith("En astrophysique, un trou noir"))
        assertEquals("Trou noir", reponse.sources.first().title)
    }

    @Test
    fun calculeEtResteHonneteHorsLigne() = runBlocking {
        engine.start()
        assertEquals("1 + 1 = 2", engine.ask("1+1", Mode.LOCAL, 1, emptyList()).text)
        assertTrue(engine.ask("yo", Mode.ANALYSE, 1, emptyList()).text.startsWith("Salut"))
        assertTrue(engine.ask("connard", Mode.LOCAL, 1, emptyList()).text.startsWith("Je reste poli"))
        val perso = engine.ask("j'aime le caca et toi?", Mode.LOCAL, 1, emptyList()).text
        assertTrue(perso, perso.startsWith("Je suis une IA d'analyse"))
        // mode de base sans la réponse sur le téléphone : recherche rapide sur internet si connecté,
        // sinon réponse honnête qui renvoie vers le mode Analyse
        val inconnu = engine.ask("C'est quoi la photosynthèse ?", Mode.LOCAL, 1, emptyList()).text
        assertTrue(inconnu, !inconnu.startsWith("Désolé, une erreur"))
        assertTrue(inconnu, inconnu.contains("photosynth", ignoreCase = true) || inconnu.contains("Analyse"))
    }

    @Test
    fun analyseNeJamaisPlanter() = runBlocking {
        engine.start()
        // Avec ou sans internet sur l'émulateur, l'analyse doit répondre proprement.
        val reponse = engine.ask("Quelle est la hauteur de la tour Eiffel ?", Mode.ANALYSE, 1, emptyList())
        assertTrue(reponse.text.isNotBlank())
        assertTrue("erreur Python : ${reponse.text}", !reponse.text.startsWith("Désolé, une erreur"))
    }
}
