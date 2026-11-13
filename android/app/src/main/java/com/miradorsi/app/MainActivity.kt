package com.miradorsi.app

import android.content.ActivityNotFoundException
import android.content.Intent
import android.os.Bundle
import android.speech.RecognizerIntent
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.KeyboardArrowRight
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.MoreVert
import androidx.compose.material.icons.filled.Person
import androidx.compose.material.icons.filled.Search
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalUriHandler
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.launch
import java.util.UUID

// ------------------------------------------------------------------ couleurs
private val Fond = Color(0xFF191A1A)
private val Carte = Color(0xFF202324)
private val Bordure = Color(0xFF2C3133)
private val Puce = Color(0xFF2A2F31)
private val PuceActive = Color(0xFF131515)
private val Sarcelle = Color(0xFF3D6B73)
private val SarcelleClaire = Color(0xFF6FA3AC)
private val Vert = Color(0xFF6FA584)
private val FondIcone = Color(0xFF1E2724)
private val Texte = Color(0xFFE9E9E6)
private val Texte2 = Color(0xFFA0A6AA)
private val Texte3 = Color(0xFF6E7579)
private val Orange = Color(0xFFE05A22)

private sealed interface Ecran {
    data object Accueil : Ecran
    data object Bibliotheque : Ecran
    data class Discussion(val id: String) : Ecran
}

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val engine = MiradorEngine(this)
        val store = ConversationStore(this)
        setContent {
            MaterialTheme(colorScheme = darkColorScheme(background = Fond, surface = Carte, primary = SarcelleClaire)) {
                MiradorApp(engine, store)
            }
        }
    }
}

@Composable
private fun MiradorApp(engine: MiradorEngine, store: ConversationStore) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val conversations = remember { mutableStateListOf<Conversation>().apply { addAll(store.load()) } }
    var ecran by remember { mutableStateOf<Ecran>(Ecran.Accueil) }
    var info by remember { mutableStateOf<EngineInfo?>(null) }
    var etat by remember { mutableStateOf("Démarrage de Mirador SI…") }
    var saisie by remember { mutableStateOf("") }
    var mode by remember { mutableStateOf(Mode.ANALYSE) }
    var profondeur by remember { mutableStateOf(2) }
    var occupe by remember { mutableStateOf(false) }
    var aPropos by remember { mutableStateOf(false) }

    LaunchedEffect(Unit) {
        etat = try {
            val i = engine.start()
            info = i
            if (i.hasModel) "${i.model} · hors ligne" else "Analyse sur internet prête"
        } catch (e: Throwable) {
            "Démarrage impossible : ${e.message}"
        }
    }

    val dictee = rememberLauncherForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        val texte = result.data?.getStringArrayListExtra(RecognizerIntent.EXTRA_RESULTS)?.firstOrNull()
        if (!texte.isNullOrBlank()) saisie = (saisie + " " + texte).trim()
    }
    fun dicter() {
        val intent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, "fr-FR")
            putExtra(RecognizerIntent.EXTRA_PROMPT, "Parle à Mirador SI")
        }
        try {
            dictee.launch(intent)
        } catch (e: ActivityNotFoundException) {
            Toast.makeText(context, "Aucune reconnaissance vocale sur cet appareil", Toast.LENGTH_SHORT).show()
        }
    }

    fun enregistrer() = store.save(conversations.toList())

    fun remplacer(c: Conversation) {
        val i = conversations.indexOfFirst { it.id == c.id }
        if (i >= 0) conversations.removeAt(i)
        conversations.add(0, c)
        enregistrer()
    }

    fun envoyer(idExistant: String?) {
        val question = saisie.trim()
        if (question.isEmpty() || occupe) return
        val existante = conversations.firstOrNull { it.id == idExistant }
        val modeUtilise = existante?.mode ?: mode
        val base = existante ?: Conversation(UUID.randomUUID().toString(), question.take(70), modeUtilise,
            emptyList(), System.currentTimeMillis())
        val historique = base.messages.zipWithNext()
            .filter { (a, b) -> a.fromUser && !b.fromUser }
            .map { (a, b) -> a.text to b.text }
        val avecQuestion = base.copy(messages = base.messages + ChatMessage(true, question),
            updatedAt = System.currentTimeMillis())
        remplacer(avecQuestion)
        ecran = Ecran.Discussion(avecQuestion.id)
        saisie = ""
        occupe = true
        scope.launch {
            val reponse = try {
                engine.ask(question, modeUtilise, profondeur, historique)
            } catch (e: Throwable) {
                Reply("Désolé, une erreur est survenue : ${e.message}", emptyList())
            }
            val actuelle = conversations.firstOrNull { it.id == avecQuestion.id } ?: avecQuestion
            remplacer(actuelle.copy(messages = actuelle.messages + ChatMessage(false, reponse.text, reponse.sources),
                updatedAt = System.currentTimeMillis()))
            occupe = false
        }
    }

    fun supprimer(c: Conversation) {
        conversations.removeAll { it.id == c.id }
        enregistrer()
        if (ecran == Ecran.Discussion(c.id)) ecran = Ecran.Accueil
    }

    BackHandler(enabled = ecran != Ecran.Accueil) { ecran = Ecran.Accueil }

    val composer: @Composable (String?) -> Unit = { id ->
        Composer(
            valeur = saisie, onValeur = { saisie = it },
            mode = conversations.firstOrNull { it.id == id }?.mode ?: mode,
            modeVerrouille = id != null,
            onMode = { mode = it }, profondeur = profondeur, onProfondeur = { profondeur = it },
            occupe = occupe, onEnvoyer = { envoyer(id) }, onDicter = { dicter() },
            onNouvelle = { saisie = ""; ecran = Ecran.Accueil },
        )
    }

    Box(Modifier.fillMaxSize().background(Fond)) {
        when (val e = ecran) {
            Ecran.Accueil -> Accueil(
                info = info, etat = etat, conversations = conversations,
                onOuvrir = { ecran = Ecran.Discussion(it.id) }, onSupprimer = { supprimer(it) },
                onAPropos = { aPropos = true }, onBibliotheque = { ecran = Ecran.Bibliotheque },
                onNouvelle = { saisie = "" }, composer = { composer(null) },
            )
            Ecran.Bibliotheque -> Bibliotheque(
                conversations = conversations, onRetour = { ecran = Ecran.Accueil },
                onOuvrir = { ecran = Ecran.Discussion(it.id) }, onSupprimer = { supprimer(it) },
            )
            is Ecran.Discussion -> {
                val c = conversations.firstOrNull { it.id == e.id }
                if (c == null) ecran = Ecran.Accueil else Discussion(
                    conversation = c, occupe = occupe, onRetour = { ecran = Ecran.Accueil },
                    onSupprimer = { supprimer(c) }, composer = { composer(c.id) },
                )
            }
        }
        if (aPropos) APropos(info, onFermer = { aPropos = false })
    }
}

// ------------------------------------------------------------------ accueil
@Composable
private fun Accueil(
    info: EngineInfo?, etat: String, conversations: List<Conversation>,
    onOuvrir: (Conversation) -> Unit, onSupprimer: (Conversation) -> Unit,
    onAPropos: () -> Unit, onBibliotheque: () -> Unit, onNouvelle: () -> Unit,
    composer: @Composable () -> Unit,
) {
    Column(Modifier.fillMaxSize().statusBarsPadding().navigationBarsPadding().imePadding()) {
        Row(Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 10.dp),
            verticalAlignment = Alignment.CenterVertically) {
            BoutonRond(Icons.Filled.Person, "À propos", onAPropos)
            Spacer(Modifier.width(12.dp))
            Text("mirador", color = Texte, fontSize = 30.sp, fontWeight = FontWeight.Medium)
            Spacer(Modifier.weight(1f))
            BoutonRond(MiradorIcons.NewChat, "Nouvelle conversation", onNouvelle)
            Spacer(Modifier.width(10.dp))
            BoutonRond(MiradorIcons.Library, "Bibliothèque", onBibliotheque)
        }
        Bandeau(etat, onAPropos)
        Box(Modifier.weight(1f).fillMaxWidth(), contentAlignment = Alignment.Center) {
            Image(MiradorIcons.tower(Sarcelle, Orange), contentDescription = info?.name ?: "Mirador SI",
                modifier = Modifier.size(96.dp))
        }
        conversations.take(5).forEach { LigneConversation(it, onOuvrir, onSupprimer) }
        Spacer(Modifier.height(8.dp))
        composer()
    }
}

@Composable
private fun Bandeau(etat: String, onClick: () -> Unit) {
    Row(
        Modifier.padding(horizontal = 16.dp, vertical = 8.dp).fillMaxWidth().clip(RoundedCornerShape(24.dp))
            .background(Brush.horizontalGradient(listOf(Color(0xFF243A3E), Color(0xFF1D2628))))
            .clickable(onClick = onClick).padding(horizontal = 18.dp, vertical = 16.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Image(MiradorIcons.tower(SarcelleClaire, Orange), null, Modifier.size(24.dp))
        Spacer(Modifier.width(12.dp))
        Text(etat, color = SarcelleClaire, fontSize = 16.sp, maxLines = 1, overflow = TextOverflow.Ellipsis,
            modifier = Modifier.weight(1f))
        Icon(Icons.AutoMirrored.Filled.KeyboardArrowRight, null, tint = SarcelleClaire)
    }
}

@Composable
private fun LigneConversation(c: Conversation, onOuvrir: (Conversation) -> Unit, onSupprimer: (Conversation) -> Unit) {
    var menu by remember { mutableStateOf(false) }
    Row(Modifier.fillMaxWidth().clickable { onOuvrir(c) }.padding(start = 16.dp, end = 4.dp, top = 6.dp, bottom = 6.dp),
        verticalAlignment = Alignment.CenterVertically) {
        Box(Modifier.size(42.dp).clip(CircleShape).background(FondIcone), contentAlignment = Alignment.Center) {
            Icon(if (c.mode == Mode.ANALYSE) Icons.Filled.Search else MiradorIcons.Laptop, null,
                tint = Vert, modifier = Modifier.size(22.dp))
        }
        Spacer(Modifier.width(16.dp))
        Text(c.title, color = Texte, fontSize = 17.sp, maxLines = 1, overflow = TextOverflow.Ellipsis,
            modifier = Modifier.weight(1f))
        Box {
            androidx.compose.material3.IconButton(onClick = { menu = true }) {
                Icon(Icons.Filled.MoreVert, "Options de la conversation", tint = Texte2)
            }
            DropdownMenu(expanded = menu, onDismissRequest = { menu = false }) {
                DropdownMenuItem(text = { Text("Supprimer") }, onClick = { menu = false; onSupprimer(c) })
            }
        }
    }
}

// ------------------------------------------------------------------ zone de saisie
@Composable
private fun Composer(
    valeur: String, onValeur: (String) -> Unit, mode: Mode, modeVerrouille: Boolean, onMode: (Mode) -> Unit,
    profondeur: Int, onProfondeur: (Int) -> Unit, occupe: Boolean,
    onEnvoyer: () -> Unit, onDicter: () -> Unit, onNouvelle: () -> Unit,
) {
    var menu by remember { mutableStateOf(false) }
    val invite = if (mode == Mode.ANALYSE) "Que souhaitez-vous analyser ?" else "Discuter avec Mirador (hors ligne)…"
    Surface(
        shape = RoundedCornerShape(30.dp), color = Carte, border = BorderStroke(1.dp, Bordure),
        modifier = Modifier.fillMaxWidth().padding(start = 12.dp, end = 12.dp, bottom = 10.dp),
    ) {
        Column(Modifier.padding(start = 18.dp, end = 12.dp, top = 16.dp, bottom = 12.dp)) {
            BasicTextField(
                value = valeur, onValueChange = onValeur,
                textStyle = TextStyle(color = Texte, fontSize = 18.sp),
                cursorBrush = SolidColor(SarcelleClaire),
                modifier = Modifier.fillMaxWidth().heightIn(min = 52.dp),
                decorationBox = { champ ->
                    Box {
                        if (valeur.isEmpty()) Text(invite, color = Texte3, fontSize = 18.sp)
                        champ()
                    }
                },
            )
            Spacer(Modifier.height(10.dp))
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box {
                    BoutonRond(Icons.Filled.Add, "Réglages de l'analyse", { menu = true }, taille = 46)
                    DropdownMenu(expanded = menu, onDismissRequest = { menu = false }) {
                        listOf(1 to "Analyse rapide", 2 to "Analyse standard", 3 to "Analyse approfondie").forEach { (n, nom) ->
                            DropdownMenuItem(
                                text = { Text((if (n == profondeur) "✓ " else "    ") + nom) },
                                onClick = { onProfondeur(n); menu = false },
                            )
                        }
                        DropdownMenuItem(text = { Text("Nouvelle conversation") }, onClick = { menu = false; onNouvelle() })
                    }
                }
                Spacer(Modifier.width(10.dp))
                Row(Modifier.clip(RoundedCornerShape(23.dp)).background(Puce).padding(3.dp)) {
                    Segment(Icons.Filled.Search, "Analyse sur internet", mode == Mode.ANALYSE, !modeVerrouille) { onMode(Mode.ANALYSE) }
                    Segment(MiradorIcons.Laptop, "Modèle local", mode == Mode.LOCAL, !modeVerrouille) { onMode(Mode.LOCAL) }
                }
                Spacer(Modifier.weight(1f))
                BoutonRond(MiradorIcons.Mic, "Dicter", onDicter, taille = 46)
                Spacer(Modifier.width(10.dp))
                Box(
                    Modifier.size(52.dp).clip(CircleShape).background(Color(0xFFEDEDEA))
                        .clickable(enabled = !occupe) { if (valeur.isBlank()) onDicter() else onEnvoyer() },
                    contentAlignment = Alignment.Center,
                ) {
                    when {
                        occupe -> CircularProgressIndicator(Modifier.size(24.dp), color = Fond, strokeWidth = 2.5.dp)
                        valeur.isBlank() -> Icon(MiradorIcons.Waveform, "Parler", tint = Fond)
                        else -> Icon(MiradorIcons.ArrowUp, "Envoyer", tint = Fond)
                    }
                }
            }
        }
    }
}

@Composable
private fun Segment(icone: ImageVector, description: String, actif: Boolean, modifiable: Boolean, onClick: () -> Unit) {
    Box(
        Modifier.size(width = 50.dp, height = 40.dp).clip(RoundedCornerShape(20.dp))
            .background(if (actif) PuceActive else Color.Transparent)
            .clickable(enabled = modifiable, onClick = onClick),
        contentAlignment = Alignment.Center,
    ) {
        Icon(icone, description, tint = if (actif) Texte else Texte2, modifier = Modifier.size(22.dp))
    }
}

@Composable
private fun BoutonRond(icone: ImageVector, description: String, onClick: () -> Unit, taille: Int = 52) {
    Box(
        Modifier.size(taille.dp).clip(CircleShape).background(Puce).border(1.dp, Bordure, CircleShape)
            .clickable(onClick = onClick),
        contentAlignment = Alignment.Center,
    ) {
        Icon(icone, description, tint = Texte, modifier = Modifier.size((taille * 0.46f).dp))
    }
}

// ------------------------------------------------------------------ discussion
@Composable
private fun Discussion(
    conversation: Conversation, occupe: Boolean, onRetour: () -> Unit, onSupprimer: () -> Unit,
    composer: @Composable () -> Unit,
) {
    val liste = rememberLazyListState()
    var menu by remember { mutableStateOf(false) }
    LaunchedEffect(conversation.messages.size, occupe) {
        val dernier = conversation.messages.size + (if (occupe) 1 else 0) - 1
        if (dernier >= 0) liste.animateScrollToItem(dernier)
    }
    Column(Modifier.fillMaxSize().statusBarsPadding().navigationBarsPadding().imePadding()) {
        Row(Modifier.fillMaxWidth().padding(horizontal = 8.dp, vertical = 6.dp), verticalAlignment = Alignment.CenterVertically) {
            androidx.compose.material3.IconButton(onClick = onRetour) {
                Icon(Icons.AutoMirrored.Filled.ArrowBack, "Retour", tint = Texte)
            }
            Text(conversation.title, color = Texte, fontSize = 18.sp, fontWeight = FontWeight.Medium,
                maxLines = 1, overflow = TextOverflow.Ellipsis, modifier = Modifier.weight(1f))
            Box {
                androidx.compose.material3.IconButton(onClick = { menu = true }) {
                    Icon(Icons.Filled.MoreVert, "Options", tint = Texte2)
                }
                DropdownMenu(expanded = menu, onDismissRequest = { menu = false }) {
                    DropdownMenuItem(text = { Text("Supprimer la conversation") }, onClick = { menu = false; onSupprimer() })
                }
            }
        }
        LazyColumn(
            state = liste, modifier = Modifier.weight(1f).fillMaxWidth(),
            contentPadding = PaddingValues(horizontal = 16.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            items(conversation.messages) { Message(it) }
            if (occupe) item { EnCours(conversation.mode) }
        }
        composer()
    }
}

@Composable
private fun Message(m: ChatMessage) {
    if (m.fromUser) {
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.End) {
            Surface(shape = RoundedCornerShape(20.dp), color = Puce, modifier = Modifier.widthIn(max = 320.dp)) {
                Text(m.text, color = Texte, fontSize = 16.sp, modifier = Modifier.padding(horizontal = 16.dp, vertical = 12.dp))
            }
        }
        return
    }
    val liens = LocalUriHandler.current
    Column(Modifier.fillMaxWidth()) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Image(MiradorIcons.tower(SarcelleClaire, Orange), null, Modifier.size(22.dp))
            Spacer(Modifier.width(8.dp))
            Text("Mirador SI", color = Texte2, fontSize = 14.sp, fontWeight = FontWeight.Medium)
        }
        Spacer(Modifier.height(6.dp))
        Text(m.text, color = Texte, fontSize = 16.sp, lineHeight = 23.sp)
        if (m.sources.isNotEmpty()) {
            Spacer(Modifier.height(10.dp))
            Text("Sources", color = Texte2, fontSize = 14.sp, fontWeight = FontWeight.Medium)
            m.sources.forEach { s ->
                Row(
                    Modifier.fillMaxWidth().padding(top = 6.dp).clip(RoundedCornerShape(12.dp)).background(Carte)
                        .clickable { runCatching { liens.openUri(s.url) } }.padding(horizontal = 12.dp, vertical = 10.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text("[${s.id}]", color = SarcelleClaire, fontSize = 14.sp, fontWeight = FontWeight.Bold)
                    Spacer(Modifier.width(10.dp))
                    Column(Modifier.weight(1f)) {
                        Text(s.title, color = Texte, fontSize = 14.sp, maxLines = 1, overflow = TextOverflow.Ellipsis)
                        Text("${s.domain} · fiabilité ${s.reliability}", color = Texte3, fontSize = 12.sp)
                    }
                }
            }
        }
    }
}

@Composable
private fun EnCours(mode: Mode) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        CircularProgressIndicator(Modifier.size(18.dp), color = SarcelleClaire, strokeWidth = 2.dp)
        Spacer(Modifier.width(12.dp))
        Text(
            if (mode == Mode.ANALYSE) "Recherche, lecture et recoupement des sources…" else "Mirador réfléchit…",
            color = Texte2, fontSize = 15.sp,
        )
    }
}

// ------------------------------------------------------------------ bibliothèque et à propos
@Composable
private fun Bibliotheque(
    conversations: List<Conversation>, onRetour: () -> Unit,
    onOuvrir: (Conversation) -> Unit, onSupprimer: (Conversation) -> Unit,
) {
    Column(Modifier.fillMaxSize().statusBarsPadding().navigationBarsPadding()) {
        Row(Modifier.fillMaxWidth().padding(horizontal = 8.dp, vertical = 6.dp), verticalAlignment = Alignment.CenterVertically) {
            androidx.compose.material3.IconButton(onClick = onRetour) {
                Icon(Icons.AutoMirrored.Filled.ArrowBack, "Retour", tint = Texte)
            }
            Text("Bibliothèque", color = Texte, fontSize = 22.sp, fontWeight = FontWeight.Medium)
        }
        if (conversations.isEmpty()) {
            Text("Aucune conversation pour l'instant.", color = Texte2, modifier = Modifier.padding(24.dp))
        }
        LazyColumn(Modifier.weight(1f)) {
            items(conversations) { LigneConversation(it, onOuvrir, onSupprimer) }
        }
    }
}

@Composable
private fun APropos(info: EngineInfo?, onFermer: () -> Unit) {
    AlertDialog(
        onDismissRequest = onFermer,
        confirmButton = { TextButton(onClick = onFermer) { Text("Fermer") } },
        icon = { Image(MiradorIcons.tower(SarcelleClaire, Orange), null, Modifier.size(48.dp)) },
        title = { Text(info?.let { "${it.name} ${it.version}" } ?: "Mirador SI") },
        text = {
            Column {
                info?.let {
                    Text(it.tagline, fontWeight = FontWeight.Medium)
                    Spacer(Modifier.height(8.dp))
                    Text(it.description)
                    Spacer(Modifier.height(8.dp))
                    Text(if (it.hasModel) "Modèle embarqué : ${it.model}" else "Aucun modèle embarqué.")
                    if (it.principles.isNotEmpty()) {
                        Spacer(Modifier.height(8.dp))
                        Text("Principes", fontWeight = FontWeight.Medium)
                        it.principles.forEach { p -> Text("• $p") }
                    }
                } ?: Text("Mirador SI démarre…")
                Spacer(Modifier.height(10.dp))
                Text(
                    "IA créée de zéro. Le modèle local a été entraîné sur des articles de Wikipédia " +
                        "(licence CC BY-SA). Les analyses lisent des sources sur internet et les citent.",
                    color = Texte2, fontSize = 13.sp,
                )
            }
        },
    )
}
