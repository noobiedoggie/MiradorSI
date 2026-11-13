package com.miradorsi.app

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

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
    fun analyseNeJamaisPlanter() = runBlocking {
        engine.start()
        // Avec ou sans internet sur l'émulateur, l'analyse doit répondre proprement.
        val reponse = engine.ask("Quelle est la hauteur de la tour Eiffel ?", Mode.ANALYSE, 1, emptyList())
        assertTrue(reponse.text.isNotBlank())
        assertTrue("erreur Python : ${reponse.text}", !reponse.text.startsWith("Désolé, une erreur"))
    }
}
