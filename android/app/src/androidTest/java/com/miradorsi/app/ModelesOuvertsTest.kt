package com.miradorsi.app

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Le moteur natif (llama.cpp) tourne vraiment sur Android : on télécharge un tout petit modèle
 * ouvert (SmolLM2 135M, 105 Mo, Apache 2.0) avec le gestionnaire de l'appli, puis on le fait écrire.
 */
@RunWith(AndroidJUnit4::class)
class ModelesOuvertsTest {
    private val context = InstrumentationRegistry.getInstrumentation().targetContext
    private val petit = ModeleOuvert(
        id = "essai-smollm2", nom = "SmolLM2 135M", editeur = "Hugging Face", role = "Essai",
        resume = "Modèle minuscule pour les tests.", depot = "bartowski/SmolLM2-135M-Instruct-GGUF",
        fichier = "SmolLM2-135M-Instruct-Q4_K_M.gguf", octets = 105_454_432, licence = "Apache 2.0", contexte = 2048,
    )

    @Test
    fun catalogueDeTroisModeles() {
        assertEquals(3, Catalogue.MODELES.size)
        assertEquals(listOf("Rapide", "Équilibré", "Puissant"), Catalogue.MODELES.map { it.role })
        Catalogue.MODELES.forEach { assertTrue(it.url, it.url.startsWith("https://huggingface.co/")) }
    }

    @Test
    fun telechargeEtFaitEcrireUnModeleOuvert() = runBlocking {
        val modeles = GestionnaireModeles(context)
        assertTrue("moteur natif indisponible", modeles.moteurDisponible())
        if (!modeles.installe(petit)) modeles.telecharger(petit) { _, _ -> }
        assertTrue(modeles.installe(petit))

        val morceaux = mutableListOf<String>()
        val texte = modeles.rediger(petit, listOf(
            "system" to "Tu es Mirador SI. Réponds brièvement.",
            "user" to "Dis bonjour en une phrase.",
        ), maxTokens = 24) { morceaux += it }
        assertTrue("texte vide", texte.isNotBlank())
        assertTrue("pas d'écriture progressive", morceaux.size >= 2)
        assertTrue(morceaux.last().trim().startsWith(morceaux.first().trim().take(3)))
        modeles.supprimer(petit)
        assertTrue(!modeles.installe(petit))
    }
}
