package com.miradorsi.app

import androidx.compose.ui.test.assertCountEquals
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.onRoot
import androidx.compose.ui.test.printToString
import androidx.compose.ui.test.hasSetTextAction
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onAllNodesWithText
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performTextInput
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

/** L'accueil contient les éléments de la maquette, et une vraie conversation s'affiche. */
@RunWith(AndroidJUnit4::class)
class InterfaceTest {
    @get:Rule
    val regle = createAndroidComposeRule<MainActivity>()

    private fun attendreTexte(texte: String, delaiMs: Long = 30_000) {
        try {
            regle.waitUntil(delaiMs) {
                regle.onAllNodesWithText(texte, substring = true).fetchSemanticsNodes().isNotEmpty()
            }
        } catch (e: Throwable) {
            // en cas d'échec, on montre ce qui est vraiment à l'écran
            throw AssertionError("« $texte » absent après $delaiMs ms. Écran : " +
                regle.onRoot(useUnmergedTree = true).printToString().replace(Regex("\\s+"), " ").take(1500), e)
        }
    }

    @Test
    fun accueilContientLesElementsDeLaMaquette() {
        attendreTexte("hors ligne")  // bandeau : le modèle embarqué est chargé
        regle.onNodeWithText("mirador").assertIsDisplayed()
        regle.onNodeWithContentDescription("À propos").assertIsDisplayed()
        regle.onNodeWithContentDescription("Mirador SI").assertIsDisplayed()  // la tour
        regle.onNodeWithText("Que souhaitez-vous analyser ?").assertIsDisplayed()
        regle.onNodeWithContentDescription("Réglages de l'analyse").assertIsDisplayed()
        regle.onNodeWithContentDescription("Analyse sur internet").assertIsDisplayed()
        regle.onNodeWithContentDescription("Modèle local").assertIsDisplayed()
        regle.onNodeWithContentDescription("Dicter").assertIsDisplayed()
        attendreTexte("Bibliothèque")  // bandeau de la bibliothèque hors ligne (à télécharger ou installée)
    }

    @Test
    fun conversationHorsLigneAfficheLaReponse() {
        attendreTexte("hors ligne")
        regle.onNodeWithContentDescription("Modèle local").performClick()
        regle.onNode(hasSetTextAction()).performTextInput("Qui es-tu ?")
        regle.onNodeWithContentDescription("Envoyer").performClick()
        attendreTexte("Je suis Mirador SI")
        // la question apparaît dans le titre de la conversation et dans la bulle de l'utilisateur
        regle.onAllNodesWithText("Qui es-tu ?").assertCountEquals(2)
    }

    @Test
    fun leBandeauOuvreLeChoixDesTroisModeles() {
        attendreTexte("paramètres")  // bandeau du modèle (Vigie)
        regle.onAllNodesWithText("paramètres", substring = true)[0].performClick()
        attendreTexte("Cerveau de Mirador")
        listOf("Vigie", "Gemma 3 1B", "Qwen3.5 2B", "Ministral 3 3B").forEach {
            regle.onAllNodesWithText(it, substring = true).fetchSemanticsNodes().isNotEmpty() ||
                throw AssertionError("$it absent du choix des modèles")
        }
        regle.onNodeWithText("Fermer").performClick()
    }
}
