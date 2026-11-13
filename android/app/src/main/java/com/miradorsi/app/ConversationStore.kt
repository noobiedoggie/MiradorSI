package com.miradorsi.app

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

data class ChatMessage(val fromUser: Boolean, val text: String, val sources: List<Source> = emptyList())

data class Conversation(
    val id: String,
    val title: String,
    val mode: Mode,
    val messages: List<ChatMessage>,
    val updatedAt: Long,
)

/** Les conversations sont gardées sur le téléphone, dans un simple fichier JSON. */
class ConversationStore(context: Context) {
    private val file = File(context.filesDir, "conversations.json")

    fun load(): List<Conversation> = try {
        if (!file.exists()) emptyList() else {
            val array = JSONArray(file.readText())
            (0 until array.length()).map { decode(array.getJSONObject(it)) }.sortedByDescending { it.updatedAt }
        }
    } catch (e: Exception) {
        emptyList()
    }

    fun save(conversations: List<Conversation>) {
        val array = JSONArray()
        conversations.forEach { array.put(encode(it)) }
        val tmp = File(file.parentFile, "conversations.tmp")
        tmp.writeText(array.toString())
        tmp.renameTo(file)
    }

    private fun encode(c: Conversation) = JSONObject().apply {
        put("id", c.id)
        put("title", c.title)
        put("mode", c.mode.key)
        put("updatedAt", c.updatedAt)
        put("messages", JSONArray().apply {
            c.messages.forEach { m ->
                put(JSONObject().apply {
                    put("fromUser", m.fromUser)
                    put("text", m.text)
                    put("sources", JSONArray().apply {
                        m.sources.forEach { s ->
                            put(JSONObject().apply {
                                put("id", s.id); put("title", s.title); put("url", s.url)
                                put("domain", s.domain); put("reliability", s.reliability)
                            })
                        }
                    })
                })
            }
        })
    }

    private fun decode(o: JSONObject): Conversation {
        val messages = o.optJSONArray("messages") ?: JSONArray()
        return Conversation(
            id = o.getString("id"),
            title = o.optString("title"),
            mode = if (o.optString("mode") == Mode.LOCAL.key) Mode.LOCAL else Mode.ANALYSE,
            updatedAt = o.optLong("updatedAt"),
            messages = (0 until messages.length()).map { i ->
                val m = messages.getJSONObject(i)
                val sources = m.optJSONArray("sources") ?: JSONArray()
                ChatMessage(
                    fromUser = m.optBoolean("fromUser"),
                    text = m.optString("text"),
                    sources = (0 until sources.length()).map { j ->
                        val s = sources.getJSONObject(j)
                        Source(s.optInt("id"), s.optString("title"), s.optString("url"),
                            s.optString("domain"), s.optString("reliability"))
                    },
                )
            },
        )
    }
}
