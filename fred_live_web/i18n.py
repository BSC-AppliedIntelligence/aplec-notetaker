"""The site's own text in English, Spanish and Catalan.

English is the source. Templates call t("English text"), and CATALOG maps it to
the other two; a string missing from the catalog shows in English; every template and app
string should have both translations.
Meeting content (notes, transcript, chat) is never translated here: notes follow
the summary language. Each user picks the interface language in Settings
(user_prefs `ui_language`); everyone else, and every page before sign-in, gets
English, which is what the site always was.

Helpers, all available in templates:
- t(text, **values): plain text, escaped by Jinja like any other value.
- th(text, **values): a translation that carries the template's own markup
  (<strong>, <a>, <code>): the catalog is trusted, the values are escaped.
- tn(singular, plural, n, **values): t() with {n}, choosing the form by n (the
  three languages share the rule: one is singular, anything else plural).
"""
from __future__ import annotations

from datetime import date, datetime

from markupsafe import Markup

from fred_live_worker.user_prefs import UI_LANGUAGES as _CODES

# code -> the language's name in itself, as the Settings picker shows it.
UI_LANGUAGES: dict[str, str] = {"en": "English", "es": "Español", "ca": "Català"}
DEFAULT_UI_LANGUAGE = "en"
assert tuple(UI_LANGUAGES) == _CODES


def translate(lang: str | None, text: str, **values) -> str:
    if lang in ("es", "ca"):
        entry = CATALOG.get(text)
        if entry:
            text = entry[0] if lang == "es" else entry[1]
    return text.format(**values) if values else text


def translate_markup(lang: str | None, text: str, **values) -> Markup:
    # Markup.format escapes every value it inserts.
    return Markup(translate(lang, text)).format(**values)


def translate_n(lang: str | None, singular: str, plural: str, n: int, **values) -> str:
    return translate(lang, singular if n == 1 else plural, n=n, **values)


# Dates, the same rules and wording as static/localtime.js, which rewrites them
# in the viewer's time zone. Python's weekday() counts from Monday.
_WEEKDAYS = {
    "en": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    "es": ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"],
    "ca": ["Dl.", "Dt.", "Dc.", "Dj.", "Dv.", "Ds.", "Dg."],
}
_MONTHS = {
    "en": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    "es": ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sept", "oct", "nov", "dic"],
    # Catalan joins day and month with "de", or "d'" before a vowel.
    "ca": ["de gen.", "de febr.", "de març", "d’abr.", "de maig", "de juny", "de jul.",
           "d’ag.", "de set.", "d’oct.", "de nov.", "de des."],
}


def format_day(day: date, today: date, lang: str | None) -> str:
    """A day as the meetings list and the meeting page label it."""
    lang = lang if lang in _WEEKDAYS else DEFAULT_UI_LANGUAGE
    delta = (today - day).days
    if delta == 0:
        return translate(lang, "Today")
    if delta == 1:
        return translate(lang, "Yesterday")
    weekday, month = _WEEKDAYS[lang][day.weekday()], _MONTHS[lang][day.month - 1]
    if lang == "en":
        return f"{weekday}, {month} {day.day}" if delta < 300 else f"{month} {day.day}, {day.year}"
    if lang == "es":
        return f"{weekday}, {day.day} {month}" if delta < 300 else f"{day.day} {month} {day.year}"
    return f"{weekday}, {day.day} {month}" if delta < 300 else f"{day.day} {month} de {day.year}"


def format_datetime_day(dt: datetime | None, lang: str | None) -> str:
    if dt is None:
        return ""
    local = dt.astimezone()
    today = datetime.now(local.tzinfo).date()
    return format_day(local.date(), today, lang)


# English -> (Spanish as written in Spain, Catalan). Grouped by page; tú/tu
# throughout, «» for quotes, the way Spanish and Catalan software writes them.
CATALOG: dict[str, tuple[str, str]] = {
    # --- shared: navigation, dates, buttons -------------------------------------
    "Aplec Notetaker home": ("Inicio de Aplec Notetaker", "Inici d'Aplec Notetaker"),
    "Users": ("Usuarios", "Usuaris"),
    "Voiceprints": ("Huellas de voz", "Empremtes de veu"),
    "Settings": ("Ajustes", "Configuració"),
    "Sign out": ("Cerrar sesión", "Tanca la sessió"),
    "Today": ("Hoy", "Avui"),
    "Yesterday": ("Ayer", "Ahir"),
    "Undated": ("Sin fecha", "Sense data"),
    "Save": ("Guardar", "Desa"),
    "Edit": ("Editar", "Edita"),
    "Delete": ("Eliminar", "Elimina"),
    "Remove": ("Quitar", "Treu"),
    "Share": ("Compartir", "Comparteix"),
    "Copy": ("Copiar", "Copia"),
    "Copied!": ("¡Copiado!", "Copiat!"),
    "Members": ("Miembros", "Membres"),

    # --- sign-in pages ------------------------------------------------------------
    "Sign in": ("Iniciar sesión", "Inicia la sessió"),
    "Welcome to Aplec": ("Te damos la bienvenida a Aplec", "Et donem la benvinguda a Aplec"),
    "Your meetings, transcribed, summarized and searchable, processed entirely on your own hardware.": (
        "Tus reuniones, transcritas, resumidas y con búsqueda, procesadas por completo en tu propio hardware.",
        "Les teves reunions, transcrites, resumides i amb cerca, processades del tot al teu propi maquinari.",
    ),
    "Continue with Google": ("Continuar con Google", "Continua amb Google"),
    "Google sign-in isn’t configured yet.": (
        "El inicio de sesión con Google aún no está configurado.",
        "L'inici de sessió amb Google encara no està configurat.",
    ),
    "Set <code>GOOGLE_OAUTH_CLIENT_ID</code> / <code>GOOGLE_OAUTH_CLIENT_SECRET</code> to enable it.": (
        "Define <code>GOOGLE_OAUTH_CLIENT_ID</code> / <code>GOOGLE_OAUTH_CLIENT_SECRET</code> para activarlo.",
        "Defineix <code>GOOGLE_OAUTH_CLIENT_ID</code> / <code>GOOGLE_OAUTH_CLIENT_SECRET</code> per activar-lo.",
    ),
    "Continue as {user}": ("Continuar como {user}", "Continua com a {user}"),
    "Aplec Notetaker · self-hosted meeting notes": (
        "Aplec Notetaker · notas de reuniones en tu propio servidor",
        "Aplec Notetaker · notes de reunions al teu propi servidor",
    ),
    "Access denied": ("Acceso denegado", "Accés denegat"),
    "Not on the list": ("No estás en la lista", "No ets a la llista"),
    "<strong>{email}</strong> isn’t allowed to access this Aplec instance.": (
        "<strong>{email}</strong> no tiene permiso para acceder a esta instancia de Aplec.",
        "<strong>{email}</strong> no té permís per accedir a aquesta instància d'Aplec.",
    ),
    "Your account isn’t allowed here.": ("Tu cuenta no tiene acceso aquí.", "El teu compte no hi té accés."),
    "Ask the administrator to add you to <code>FRED_LIVE_WEB_USERS</code>.": (
        "Pide al administrador que te añada a <code>FRED_LIVE_WEB_USERS</code>.",
        "Demana a l'administrador que t'afegeixi a <code>FRED_LIVE_WEB_USERS</code>.",
    ),
    "Try another account": ("Probar con otra cuenta", "Prova amb un altre compte"),
    "Access requested": ("Acceso solicitado", "Accés sol·licitat"),
    "Request sent": ("Solicitud enviada", "Sol·licitud enviada"),
    "<strong>{email}</strong> is now waiting for approval.": (
        "<strong>{email}</strong> está pendiente de aprobación.",
        "<strong>{email}</strong> està pendent d'aprovació.",
    ),
    "An administrator has to let you in once; after that, signing in just works.": (
        "Un administrador tiene que darte acceso una vez; después, podrás iniciar sesión sin más.",
        "Un administrador t'ha de donar accés una vegada; després, podràs iniciar la sessió sense més.",
    ),
    "Give them a nudge if you're in a hurry.": (
        "Si tienes prisa, avísale.",
        "Si tens pressa, avisa'l.",
    ),
    "Sign in again": ("Volver a iniciar sesión", "Torna a iniciar la sessió"),
    "Not found": ("No encontrada", "No s'ha trobat"),
    "Meeting not found": ("Reunión no encontrada", "No s'ha trobat la reunió"),
    "This meeting doesn’t exist or belongs to another user.": (
        "Esta reunión no existe o pertenece a otro usuario.",
        "Aquesta reunió no existeix o és d'un altre usuari.",
    ),
    "Back to meetings": ("Volver a las reuniones", "Torna a les reunions"),

    # --- onboarding ---------------------------------------------------------------
    "Welcome": ("Bienvenida", "Benvinguda"),
    "Welcome, {name}!": ("¡Hola, {name}!", "Hola, {name}!"),
    "One step left: connect your Google Calendar and Aplec will join your meetings automatically, record them, and turn them into notes, action items and answers.": (
        "Solo falta un paso: conecta tu Google Calendar y Aplec se unirá a tus reuniones automáticamente, las grabará y las convertirá en notas, tareas y respuestas.",
        "Només falta un pas: connecta el teu Google Calendar i Aplec s'unirà a les teves reunions automàticament, les gravarà i les convertirà en notes, tasques i respostes.",
    ),
    "<strong>Connect your calendar</strong>: Aplec sees your upcoming Google Meet events": (
        "<strong>Conecta tu calendario</strong>: Aplec ve tus próximos eventos de Google Meet",
        "<strong>Connecta el teu calendari</strong>: Aplec veu els teus propers esdeveniments de Google Meet",
    ),
    "<strong>Aplec joins by itself</strong>: invited &amp; signed in, moments before each meeting starts": (
        "<strong>Aplec se une solo</strong>: invitado y con sesión iniciada, momentos antes de que empiece cada reunión",
        "<strong>Aplec s'hi uneix sol</strong>: convidat i amb la sessió iniciada, uns moments abans que comenci cada reunió",
    ),
    "<strong>Notes appear here</strong>: summary, action items, transcript, and a chat that answers from the meeting": (
        "<strong>Las notas aparecen aquí</strong>: resumen, tareas, transcripción y un chat que responde a partir de la reunión",
        "<strong>Les notes apareixen aquí</strong>: resum, tasques, transcripció i un xat que respon a partir de la reunió",
    ),
    "Which meetings should Aplec join?": ("¿A qué reuniones debe unirse Aplec?", "A quines reunions s'ha d'unir Aplec?"),
    "Language for your meeting notes": ("Idioma de las notas de tus reuniones", "Idioma de les notes de les teves reunions"),
    "Connect Google Calendar": ("Conectar Google Calendar", "Connecta Google Calendar"),
    "Skip for now": ("Ahora no", "Ara no"),

    # --- meetings list ------------------------------------------------------------
    "Meetings": ("Reuniones", "Reunions"),
    "{n} meeting": ("{n} reunión", "{n} reunió"),
    "{n} meetings": ("{n} reuniones", "{n} reunions"),
    "{n} meeting matching “{q}”": ("{n} reunión que coincide con «{q}»", "{n} reunió que coincideix amb «{q}»"),
    "{n} meetings matching “{q}”": ("{n} reuniones que coinciden con «{q}»", "{n} reunions que coincideixen amb «{q}»"),
    "Search meetings…": ("Buscar reuniones…", "Cerca reunions…"),
    "Calendar connected. Aplec will join your upcoming Google Meet events automatically and their notes will appear here.": (
        "Calendario conectado. Aplec se unirá automáticamente a tus próximos eventos de Google Meet y sus notas aparecerán aquí.",
        "Calendari connectat. Aplec s'unirà automàticament als teus propers esdeveniments de Google Meet i les seves notes apareixeran aquí.",
    ),
    "<strong>Connect your Google Calendar</strong> so Aplec can join your meetings automatically and take notes.": (
        "<strong>Conecta tu Google Calendar</strong> para que Aplec se una a tus reuniones automáticamente y tome notas.",
        "<strong>Connecta el teu Google Calendar</strong> perquè Aplec s'uneixi a les teves reunions automàticament i prengui notes.",
    ),
    "Connect calendar": ("Conectar calendario", "Connecta el calendari"),
    "Filter by platform": ("Filtrar por plataforma", "Filtra per plataforma"),
    "All": ("Todas", "Totes"),
    "No matches": ("Sin resultados", "Cap resultat"),
    "No meetings match “{q}” on {platform}.": (
        "Ninguna reunión de {platform} coincide con «{q}».",
        "Cap reunió de {platform} no coincideix amb «{q}».",
    ),
    "No meetings match “{q}”.": ("Ninguna reunión coincide con «{q}».", "Cap reunió no coincideix amb «{q}»."),
    "No meetings on {platform}.": ("No hay reuniones de {platform}.", "No hi ha reunions de {platform}."),
    "Try clearing the filter.": ("Prueba a quitar el filtro.", "Prova de treure el filtre."),
    "No meetings yet": ("Aún no hay reuniones", "Encara no hi ha reunions"),
    "Aplec joins your Google Calendar meetings automatically and the notes appear here minutes after the call ends.": (
        "Aplec se une automáticamente a las reuniones de tu Google Calendar y las notas aparecen aquí pocos minutos después de que termine la llamada.",
        "Aplec s'uneix automàticament a les reunions del teu Google Calendar i les notes apareixen aquí pocs minuts després que acabi la trucada.",
    ),
    "Shared": ("Compartida", "Compartida"),
    "Processing": ("Procesando", "Processant"),
    "Aplec is transcribing and summarizing this meeting. It appears here automatically.": (
        "Aplec está transcribiendo y resumiendo esta reunión. Aparecerá aquí automáticamente.",
        "Aplec està transcrivint i resumint aquesta reunió. Apareixerà aquí automàticament.",
    ),
    "Failed": ("Error", "Error"),
    "Reprocess": ("Reprocesar", "Reprocessa"),
    "View last good result": ("Ver el último resultado correcto", "Mostra l'últim resultat correcte"),
    "Processed": ("Procesada", "Processada"),
    "Needs review": ("Revisar", "Cal revisar"),
    "No speech": ("Sin voz", "Sense veu"),

    # --- meeting page -------------------------------------------------------------
    "Meeting {id}": ("Reunión {id}", "Reunió {id}"),
    "All meetings": ("Todas las reuniones", "Totes les reunions"),
    "Copy summary, key points, action items, decisions and risks — ready to paste into an email": (
        "Copia el resumen, los puntos clave, las tareas, las decisiones y los riesgos, listos para pegar en un correo",
        "Copia el resum, els punts clau, les tasques, les decisions i els riscos, a punt per enganxar en un correu",
    ),
    "Copy notes": ("Copiar notas", "Copia les notes"),
    "Copy failed": ("No se ha podido copiar", "No s'ha pogut copiar"),
    "Transcript": ("Transcripción", "Transcripció"),
    "No speech was detected in this recording. Aplec was probably alone in the call.": (
        "No se ha detectado voz en esta grabación. Probablemente Aplec estaba solo en la llamada.",
        "No s'ha detectat veu en aquesta gravació. Probablement Aplec era sol a la trucada.",
    ),
    "Aplec could not post the recording notice in the meeting chat ({reason}).": (
        "Aplec no ha podido publicar el aviso de grabación en el chat de la reunión ({reason}).",
        "Aplec no ha pogut publicar l'avís de gravació al xat de la reunió ({reason}).",
    ),
    "Aplec could not post the recording notice in the meeting chat.": (
        "Aplec no ha podido publicar el aviso de grabación en el chat de la reunión.",
        "Aplec no ha pogut publicar l'avís de gravació al xat de la reunió.",
    ),
    "This meeting’s page had no chat Aplec could post the recording notice to.": (
        "La página de esta reunión no tenía un chat donde Aplec pudiera publicar el aviso de grabación.",
        "La pàgina d'aquesta reunió no tenia cap xat on Aplec pogués publicar l'avís de gravació.",
    ),
    "Participants may not have been told.": (
        "Puede que los participantes no lo sepan.",
        "Potser els participants no ho saben.",
    ),
    "Summary": ("Resumen", "Resum"),
    "Key points": ("Puntos clave", "Punts clau"),
    "Action items": ("Tareas", "Tasques"),
    "Decisions": ("Decisiones", "Decisions"),
    "Risks & open questions": ("Riesgos y preguntas abiertas", "Riscos i preguntes obertes"),
    "(due {date})": ("(para el {date})", "(per al {date})"),
    "Edit speakers": ("Editar hablantes", "Edita els parlants"),
    'Fix hybrid-room labels: rename a <span class="mono">SPEAKER_N</span> to a real name, or reassign one name to another (e.g. when several people shared one mic). Applies everywhere: transcript, action-item owners, and name mentions in the summary and notes.': (
        'Corrige las etiquetas de las salas híbridas: cambia un <span class="mono">SPEAKER_N</span> por un nombre real o reasigna un nombre a otro (p. ej., cuando varias personas compartían un micrófono). Se aplica en todas partes: transcripción, responsables de tareas y menciones de nombres en el resumen y las notas.',
        'Corregeix les etiquetes de les sales híbrides: canvia un <span class="mono">SPEAKER_N</span> per un nom real o reassigna un nom a un altre (p. ex., quan diverses persones compartien un micròfon). S\'aplica a tot arreu: transcripció, responsables de les tasques i mencions de noms al resum i a les notes.',
    ),
    "Save names": ("Guardar nombres", "Desa els noms"),
    "Access": ("Acceso", "Accés"),
    "You can always see this meeting.": ("Siempre puedes ver esta reunión.", "Sempre pots veure aquesta reunió."),
    "You and the admin can always see this meeting.": (
        "Tú y el administrador siempre podéis ver esta reunión.",
        "Tu i l'administrador sempre podeu veure aquesta reunió.",
    ),
    "People below were given read access (notes, transcript, ask): they cannot reprocess, rename speakers, or re-share. Team shares add the team's <em>current</em> members; someone joining the team later does not gain access to this meeting. Colleagues Aplec recognized by voice are added automatically (turn that off in Settings); anyone you remove stays removed.": (
        "Las personas de abajo tienen acceso de lectura (notas, transcripción y preguntas): no pueden reprocesar, renombrar hablantes ni volver a compartir. Compartir con un equipo añade a sus miembros <em>actuales</em>; quien se una al equipo más tarde no tendrá acceso a esta reunión. Los compañeros que Aplec reconoce por la voz se añaden automáticamente (puedes desactivarlo en Ajustes); a quien quites, no se le vuelve a añadir.",
        "Les persones de sota tenen accés de lectura (notes, transcripció i preguntes): no poden reprocessar, canviar el nom dels parlants ni tornar a compartir. Compartir amb un equip afegeix els seus membres <em>actuals</em>; qui s'uneixi a l'equip més endavant no tindrà accés a aquesta reunió. Els companys que Aplec reconeix per la veu s'afegeixen automàticament (ho pots desactivar a Configuració); a qui treguis, no se'l torna a afegir.",
    ),
    "recognized voice": ("voz reconocida", "veu reconeguda"),
    "Shared {date} by {who}": ("Compartida el {date} por {who}", "Compartida el {date} per {who}"),
    "Not shared with anyone yet.": ("Aún no se ha compartido con nadie.", "Encara no s'ha compartit amb ningú."),
    "Share with a person…": ("Compartir con una persona…", "Comparteix amb una persona…"),
    "Share with a team…": ("Compartir con un equipo…", "Comparteix amb un equip…"),
    "Shared with you by <strong>{who}</strong> — read access.": (
        "Compartida contigo por <strong>{who}</strong>, con acceso de lectura.",
        "Compartida amb tu per <strong>{who}</strong>, amb accés de lectura.",
    ),
    "Shared with you — read access.": (
        "Compartida contigo, con acceso de lectura.",
        "Compartida amb tu, amb accés de lectura.",
    ),
    "Links shared in the chat": ("Enlaces compartidos en el chat", "Enllaços compartits al xat"),
    "Meeting chat · {n} message": ("Chat de la reunión · {n} mensaje", "Xat de la reunió · {n} missatge"),
    "Meeting chat · {n} messages": ("Chat de la reunión · {n} mensajes", "Xat de la reunió · {n} missatges"),
    "Transcript · {n} turn": ("Transcripción · {n} intervención", "Transcripció · {n} intervenció"),
    "Transcript · {n} turns": ("Transcripción · {n} intervenciones", "Transcripció · {n} intervencions"),
    "Ask Aplec": ("Pregunta a Aplec", "Pregunta a Aplec"),
    "Questions are answered strictly from this meeting.": (
        "Las respuestas salen solo de esta reunión.",
        "Les respostes surten només d'aquesta reunió.",
    ),
    "What were the action items, and who owns each one?": (
        "¿Cuáles fueron las tareas y quién se encarga de cada una?",
        "Quines van ser les tasques i qui s'encarrega de cadascuna?",
    ),
    "What was decided in this meeting?": ("¿Qué se decidió en esta reunión?", "Què es va decidir en aquesta reunió?"),
    "Summarize this meeting in one short paragraph.": (
        "Resume esta reunión en un párrafo breve.",
        "Resumeix aquesta reunió en un paràgraf breu.",
    ),
    "TL;DR": ("En resumen", "En resum"),
    "Thinking…": ("Pensando…", "Pensant…"),
    "e.g. What did we decide?": ("p. ej., ¿qué decidimos?", "p. ex., què vam decidir?"),
    "Send": ("Enviar", "Envia"),
    "The assistant is unavailable right now — is the worker/LLM running?": (
        "El asistente no está disponible ahora mismo. ¿Está en marcha el worker o el LLM?",
        "L'assistent no està disponible ara mateix. Funciona el worker o l'LLM?",
    ),
    "Session expired — reload the page": (
        "La sesión ha caducado. Recarga la página",
        "La sessió ha caducat. Torna a carregar la pàgina",
    ),
    "Only the owner can share": ("Solo el propietario puede compartir", "Només el propietari pot compartir"),

    # --- settings -----------------------------------------------------------------
    "Calendar connected. Aplec now joins your meetings automatically.": (
        "Calendario conectado. Ahora Aplec se une a tus reuniones automáticamente.",
        "Calendari connectat. Ara Aplec s'uneix a les teves reunions automàticament.",
    ),
    "Google didn’t issue a long-lived token. Try connecting again, and complete the consent screen fully.": (
        "Google no ha emitido un token de larga duración. Vuelve a conectar y completa la pantalla de consentimiento entera.",
        "Google no ha emès un token de llarga durada. Torna a connectar i completa la pantalla de consentiment sencera.",
    ),
    "Connected as <strong>{account}</strong>": ("Conectado como <strong>{account}</strong>", "Connectat com a <strong>{account}</strong>"),
    "Aplec watches this calendar, invites the notetaker to your Google Meet events and joins them automatically a moment before they start.": (
        "Aplec vigila este calendario, invita al notetaker a tus eventos de Google Meet y se une automáticamente un momento antes de que empiecen.",
        "Aplec vigila aquest calendari, convida el notetaker als teus esdeveniments de Google Meet i s'hi uneix automàticament un moment abans que comencin.",
    ),
    "Disconnect": ("Desconectar", "Desconnecta"),
    "Not connected": ("Sin conectar", "Sense connectar"),
    "Connect your Google Calendar so Aplec can join your meetings automatically and take notes. You’ll be asked to allow access to your calendar events, which is all Aplec needs.": (
        "Conecta tu Google Calendar para que Aplec se una a tus reuniones automáticamente y tome notas. Se te pedirá acceso a los eventos de tu calendario, que es todo lo que Aplec necesita.",
        "Connecta el teu Google Calendar perquè Aplec s'uneixi a les teves reunions automàticament i prengui notes. Se't demanarà accés als esdeveniments del teu calendari, que és tot el que Aplec necessita.",
    ),
    "Google OAuth is not configured": ("Google OAuth no está configurado", "Google OAuth no està configurat"),
    "Auto-join": ("Entrada automática", "Entrada automàtica"),
    "Which meetings Aplec joins by itself.": ("A qué reuniones entra Aplec por su cuenta.", "A quines reunions entra Aplec pel seu compte."),
    "“Meetings I organized or accepted” keeps Aplec out of team invites you haven’t RSVP’d to. “Never” turns automatic joining off entirely: use the browser extension or invite the bot manually.": (
        "«Reuniones que he organizado o aceptado» deja a Aplec fuera de las invitaciones de equipo que no has respondido. «Nunca» desactiva por completo la entrada automática: usa la extensión del navegador o invita al bot a mano.",
        "«Reunions que he organitzat o acceptat» deixa Aplec fora de les invitacions d'equip que no has respost. «Mai» desactiva del tot l'entrada automàtica: fes servir l'extensió del navegador o convida el bot a mà.",
    ),
    "Auto-join policy": ("Política de entrada automática", "Política d'entrada automàtica"),
    "Meetings I organized or accepted": ("Reuniones que he organizado o aceptado", "Reunions que he organitzat o acceptat"),
    "Every meeting with a Meet link": ("Todas las reuniones con enlace de Meet", "Totes les reunions amb enllaç de Meet"),
    "Never join automatically": ("Nunca entrar automáticamente", "No hi entris mai automàticament"),
    "Interface language": ("Idioma de la interfaz", "Idioma de la interfície"),
    "The language of Aplec's own pages: menus, titles and buttons.": (
        "El idioma de las páginas de Aplec: menús, títulos y botones.",
        "L'idioma de les pàgines d'Aplec: menús, títols i botons.",
    ),
    "Meeting notes keep following the summary language below.": (
        "Las notas de las reuniones siguen el idioma del resumen, más abajo.",
        "Les notes de les reunions segueixen l'idioma del resum, més avall.",
    ),
    "Summary language": ("Idioma del resumen", "Idioma del resum"),
    "Meeting notes are written in this language.": (
        "Las notas de las reuniones se escriben en este idioma.",
        "Les notes de les reunions s'escriuen en aquest idioma.",
    ),
    "“Same as the meeting” follows whatever language dominates each meeting; picking a fixed language translates every summary into it.": (
        "«El de la reunión» sigue el idioma que predomina en cada reunión; si eliges un idioma fijo, todos los resúmenes se traducen a él.",
        "«El de la reunió» segueix l'idioma que predomina en cada reunió; si tries un idioma fix, tots els resums s'hi tradueixen.",
    ),
    "Same as the meeting (auto)": ("El de la reunión (automático)", "El de la reunió (automàtic)"),
    "Browser extension": ("Extensión del navegador", "Extensió del navegador"),
    "Record meetings without the bot, straight from your Meet tab.": (
        "Graba reuniones sin el bot, directamente desde tu pestaña de Meet.",
        "Grava reunions sense el bot, directament des de la teva pestanya de Meet.",
    ),
    '<a href="{url}" target="_blank" rel="noopener"><strong>Install from the Chrome Web Store</strong></a> (updates arrive automatically), then paste this token and the worker URL into its options, just once. Manual install for development: <a href="/static/fred-extension.zip" download>zip</a>.': (
        '<a href="{url}" target="_blank" rel="noopener"><strong>Instálala desde la Chrome Web Store</strong></a> (las actualizaciones llegan solas) y pega este token y la URL del worker en sus opciones, solo una vez. Instalación manual para desarrollo: <a href="/static/fred-extension.zip" download>zip</a>.',
        '<a href="{url}" target="_blank" rel="noopener"><strong>Instal·la-la des de la Chrome Web Store</strong></a> (les actualitzacions arriben soles) i enganxa aquest token i la URL del worker a les seves opcions, només una vegada. Instal·lació manual per a desenvolupament: <a href="/static/fred-extension.zip" download>zip</a>.',
    ),
    '<a href="/static/fred-extension.zip" download><strong>Download the extension (.zip)</strong></a>, unzip it, load it via <span class="mono">chrome://extensions</span> → Developer mode → “Load unpacked”, then paste this token and the worker URL into its options, just once.': (
        '<a href="/static/fred-extension.zip" download><strong>Descarga la extensión (.zip)</strong></a>, descomprímela, cárgala desde <span class="mono">chrome://extensions</span> → Modo de desarrollador → «Cargar descomprimida» y pega este token y la URL del worker en sus opciones, solo una vez.',
        '<a href="/static/fred-extension.zip" download><strong>Baixa l\'extensió (.zip)</strong></a>, descomprimeix-la, carrega-la des de <span class="mono">chrome://extensions</span> → Mode de desenvolupador → «Carrega la descomprimida» i enganxa aquest token i la URL del worker a les seves opcions, només una vegada.',
    ),
    "Capture token": ("Token de captura", "Token de captura"),
    "Worker URL": ("URL del worker", "URL del worker"),
    "Voiceprint": ("Huella de voz", "Empremta de veu"),
    "Voiceprint enrolled. Aplec can now name you in meeting notes by voice.": (
        "Huella de voz registrada. Ahora Aplec puede ponerte nombre en las notas por la voz.",
        "Empremta de veu registrada. Ara Aplec et pot posar el nom a les notes per la veu.",
    ),
    "Enrolled as <strong>{name}</strong>": ("Registrada como <strong>{name}</strong>", "Registrada com a <strong>{name}</strong>"),
    "{n}s of audio on file, stored only as a numeric voice template on this server. Delete it anytime; deletion is immediate.": (
        "{n} s de audio guardados solo como una plantilla numérica de voz en este servidor. Elimínala cuando quieras; se borra al instante.",
        "{n} s d'àudio desats només com una plantilla numèrica de veu en aquest servidor. Elimina-la quan vulguis; s'esborra a l'instant.",
    ),
    "Delete your voiceprint? Aplec will stop naming you by voice.": (
        "¿Eliminar tu huella de voz? Aplec dejará de reconocerte por la voz.",
        "Vols eliminar la teva empremta de veu? Aplec deixarà de reconèixer-te per la veu.",
    ),
    "Delete my voiceprint": ("Eliminar mi huella de voz", "Elimina la meva empremta de veu"),
    "Re-record": ("Volver a grabar", "Torna a gravar"),
    "Replace your voiceprint with a fresh recording.": (
        "Sustituye tu huella de voz por una grabación nueva.",
        "Substitueix la teva empremta de veu per una gravació nova.",
    ),
    "Your team leader invited you to enroll a voiceprint so meeting notes can name you automatically.": (
        "Tu responsable te ha invitado a registrar tu huella de voz para que las notas de las reuniones te nombren automáticamente.",
        "El teu responsable t'ha convidat a registrar la teva empremta de veu perquè les notes de les reunions et posin el nom automàticament.",
    ),
    "Record about 45 seconds, or upload an audio file of you speaking alone.": (
        "Graba unos 45 segundos o sube un archivo de audio en el que hables tú solo.",
        "Grava uns 45 segons o puja un fitxer d'àudio on parlis tu sol.",
    ),
    "Decline (nothing changes for you)": ("Rechazar (para ti no cambia nada)", "Rebutja (per a tu no canvia res)"),
    "No voiceprint": ("Sin huella de voz", "Sense empremta de veu"),
    "Optional: enroll your voice so Aplec can name you in notes even when several people share one microphone.": (
        "Opcional: registra tu voz para que Aplec te nombre en las notas incluso cuando varias personas comparten un micrófono.",
        "Opcional: registra la teva veu perquè Aplec et posi el nom a les notes fins i tot quan diverses persones comparteixen un micròfon.",
    ),
    "Enroll now": ("Registrarla ahora", "Registra-la ara"),
    "What you are agreeing to.": ("A qué das tu consentimiento.", "A què dones el teu consentiment."),
    "Aplec converts your recording into a numeric voice template used only to label your name in this team's meeting notes and to give you access to the meetings where your voice is recognized (their owners can turn that off). The raw audio is deleted immediately after processing; your voice cannot be reconstructed from the template. It stays on this server, is never shared with anyone, and is deleted the moment you click delete or leave the team. Declining or deleting has no consequence: meetings and notes keep working, you simply appear by your Meet name or as SPEAKER_N.": (
        "Aplec convierte tu grabación en una plantilla numérica de voz que solo se usa para poner tu nombre en las notas de las reuniones de este equipo y para darte acceso a las reuniones en las que se reconozca tu voz (sus propietarios pueden desactivarlo). El audio original se elimina justo después de procesarlo; tu voz no se puede reconstruir a partir de la plantilla. La plantilla se queda en este servidor, nunca se comparte con nadie y se elimina en cuanto pulses eliminar o dejes el equipo. Rechazar o eliminar no tiene ninguna consecuencia: las reuniones y las notas siguen funcionando, simplemente apareces con tu nombre de Meet o como SPEAKER_N.",
        "Aplec converteix la teva gravació en una plantilla numèrica de veu que només s'utilitza per posar el teu nom a les notes de les reunions d'aquest equip i per donar-te accés a les reunions on es reconegui la teva veu (els seus propietaris ho poden desactivar). L'àudio original s'elimina just després de processar-lo; la teva veu no es pot reconstruir a partir de la plantilla. La plantilla es queda en aquest servidor, mai no es comparteix amb ningú i s'elimina en el moment que facis clic a eliminar o deixis l'equip. Rebutjar-ho o eliminar-la no té cap conseqüència: les reunions i les notes continuen funcionant, simplement apareixes amb el teu nom de Meet o com a SPEAKER_N.",
    ),
    "Name as it appears in Google Meet": ("Nombre tal como aparece en Google Meet", "Nom tal com apareix a Google Meet"),
    "● Record": ("● Grabar", "● Grava"),
    "or upload:": ("o sube un archivo:", "o puja un fitxer:"),
    "I have read the notice above and consent to the processing of my voice template.": (
        "He leído el aviso anterior y doy mi consentimiento para el tratamiento de mi plantilla de voz.",
        "He llegit l'avís anterior i dono el meu consentiment per al tractament de la meva plantilla de veu.",
    ),
    "Enroll": ("Registrar", "Registra"),
    "● Re-record ({n}s captured)": ("● Volver a grabar ({n} s grabados)", "● Torna a gravar ({n} s gravats)"),
    "■ Stop": ("■ Parar", "■ Atura"),
    "(aim for 30-60s)": ("(lo ideal son 30-60 s)", "(l'ideal són 30-60 s)"),
    "Microphone unavailable: {error}. Use the file upload instead.": (
        "Micrófono no disponible: {error}. Sube un archivo en su lugar.",
        "Micròfon no disponible: {error}. Puja un fitxer en lloc seu.",
    ),
    "Record something or choose a file first.": (
        "Graba algo o elige un archivo primero.",
        "Grava alguna cosa o tria un fitxer primer.",
    ),
    "Processing (this can take ~20s the first time)...": (
        "Procesando (la primera vez puede tardar unos 20 s)…",
        "Processant (la primera vegada pot trigar uns 20 s)…",
    ),
    "Enrollment failed: {reason}": ("No se ha podido registrar: {reason}", "No s'ha pogut registrar: {reason}"),
    "Enrollment failed (is the worker running? was the audio valid?)": (
        "No se ha podido registrar (¿está en marcha el worker?, ¿el audio era válido?)",
        "No s'ha pogut registrar (funciona el worker?, l'àudio era vàlid?)",
    ),
    "Name is required": ("El nombre es obligatorio", "El nom és obligatori"),
    "Audio too large": ("El audio es demasiado grande", "L'àudio és massa gran"),
    "Delete failed; try again": ("No se ha podido eliminar; inténtalo de nuevo", "No s'ha pogut eliminar; torna-ho a provar"),
    "Automatic sharing": ("Compartir automáticamente", "Compartició automàtica"),
    "Share the meetings I record with the colleagues Aplec recognizes in them": (
        "Compartir las reuniones que grabo con los compañeros que Aplec reconoce en ellas",
        "Compartir les reunions que gravo amb els companys que Aplec hi reconeix",
    ),
    "Read access (notes, transcript, Ask) for people with an Aplec account whose enrolled voice Aplec recognizes in the recording. Remove anyone from a meeting's Access panel and Aplec will not add them back.": (
        "Acceso de lectura (notas, transcripción y preguntas) para las personas con cuenta en Aplec cuya voz registrada Aplec reconoce en la grabación. Si quitas a alguien desde el panel de Acceso de una reunión, Aplec no volverá a añadirlo.",
        "Accés de lectura (notes, transcripció i preguntes) per a les persones amb compte a Aplec la veu registrada de les quals Aplec reconeix a la gravació. Si treus algú des del tauler d'Accés d'una reunió, Aplec no el tornarà a afegir.",
    ),
    "Account": ("Cuenta", "Compte"),
    "Signed in as <strong>{name}</strong> ({email})": (
        "Sesión iniciada como <strong>{name}</strong> ({email})",
        "Sessió iniciada com a <strong>{name}</strong> ({email})",
    ),
    "Access is managed by the instance administrator via the allowlist.": (
        "El acceso lo gestiona el administrador de la instancia mediante la lista de usuarios permitidos.",
        "L'accés el gestiona l'administrador de la instància mitjançant la llista d'usuaris permesos.",
    ),

    # --- admin: users and teams ---------------------------------------------------
    "Approve access requests and manage who can use this Aplec": (
        "Aprueba solicitudes de acceso y gestiona quién puede usar este Aplec",
        "Aprova sol·licituds d'accés i gestiona qui pot fer servir aquest Aplec",
    ),
    "Waiting for approval": ("Pendientes de aprobación", "Pendents d'aprovació"),
    "These people signed in with Google but aren't members yet. Approving runs the full onboarding (Vexa account, API key, webhook, roster) — their next sign-in just works. Rejecting deletes the request; they can re-request by signing in again.": (
        "Estas personas han iniciado sesión con Google pero aún no son miembros. Aprobar ejecuta el alta completa (cuenta de Vexa, clave de API, webhook y lista de miembros), y su próximo inicio de sesión ya funciona. Rechazar elimina la solicitud; pueden volver a pedirlo iniciando sesión de nuevo.",
        "Aquestes persones han iniciat la sessió amb Google però encara no són membres. Aprovar fa l'alta completa (compte de Vexa, clau d'API, webhook i llista de membres), i el seu proper inici de sessió ja funciona. Rebutjar elimina la sol·licitud; poden tornar-ho a demanar iniciant la sessió de nou.",
    ),
    "Requested {date}": ("Solicitado el {date}", "Sol·licitat el {date}"),
    "Approve": ("Aprobar", "Aprova"),
    "Reject": ("Rechazar", "Rebutja"),
    'Removing someone revokes their session, capture token and calendar auto-join on the next request; their Vexa account and meeting history are kept. Members configured in <span class="mono">.env.worker</span> are managed there, not here.': (
        'Quitar a alguien revoca su sesión, su token de captura y la entrada automática de su calendario en la siguiente petición; su cuenta de Vexa y su historial de reuniones se conservan. Los miembros configurados en <span class="mono">.env.worker</span> se gestionan allí, no aquí.',
        'Treure algú revoca la seva sessió, el seu token de captura i l\'entrada automàtica del seu calendari a la petició següent; el seu compte de Vexa i el seu historial de reunions es conserven. Els membres configurats a <span class="mono">.env.worker</span> es gestionen allà, no aquí.',
    ),
    "Vexa user {id}": ("Usuario de Vexa {id}", "Usuari de Vexa {id}"),
    "Remove {email}? Their meetings are kept; access stops immediately.": (
        "¿Quitar a {email}? Sus reuniones se conservan; el acceso se corta al momento.",
        "Vols treure {email}? Les seves reunions es conserven; l'accés es talla a l'instant.",
    ),
    "Teams": ("Equipos", "Equips"),
    "Groups for sharing meetings. Sharing a meeting with a team adds its members <em>at that moment</em> (snapshot): people who join the team later do not see previously shared meetings. Saving a team with an existing name replaces its member list.": (
        "Grupos para compartir reuniones. Compartir una reunión con un equipo añade a sus miembros <em>en ese momento</em>: quien se una al equipo más tarde no verá las reuniones compartidas antes. Guardar un equipo con un nombre que ya existe sustituye su lista de miembros.",
        "Grups per compartir reunions. Compartir una reunió amb un equip hi afegeix els seus membres <em>en aquell moment</em>: qui s'uneixi a l'equip més endavant no veurà les reunions compartides abans. Desar un equip amb un nom que ja existeix substitueix la seva llista de membres.",
    ),
    "{n} member": ("{n} miembro", "{n} membre"),
    "{n} members": ("{n} miembros", "{n} membres"),
    "no members": ("sin miembros", "sense membres"),
    "Delete team {name}? Existing shares stay (they are per-person).": (
        "¿Eliminar el equipo {name}? Lo que ya se ha compartido se mantiene (se comparte por persona).",
        "Vols eliminar l'equip {name}? El que ja s'ha compartit es manté (es comparteix per persona).",
    ),
    "New team name": ("Nombre del nuevo equipo", "Nom del nou equip"),
    "Save team": ("Guardar equipo", "Desa l'equip"),
    "Type an email…": ("Escribe un correo…", "Escriu un correu…"),
    "Approval failed: {detail}": ("No se ha podido aprobar: {detail}", "No s'ha pogut aprovar: {detail}"),
    "You cannot remove yourself": ("No puedes quitarte a ti mismo", "No et pots treure a tu mateix"),
    "Env-configured users are removed in .env.worker": (
        "Los usuarios configurados por entorno se quitan en .env.worker",
        "Els usuaris configurats per entorn es treuen a .env.worker",
    ),
    "Remove failed; try again": ("No se ha podido quitar; inténtalo de nuevo", "No s'ha pogut treure; torna-ho a provar"),

    # --- admin: voiceprints -------------------------------------------------------
    "Team voiceprints": ("Huellas de voz del equipo", "Empremtes de veu de l'equip"),
    "Invite members to enroll their voice for speaker identification": (
        "Invita a los miembros a registrar su voz para identificar a los hablantes",
        "Convida els membres a registrar la seva veu per identificar els parlants",
    ),
    "Each person records their own sample behind a consent notice; you can invite, cancel, or delete, but never upload audio for someone else. Declining costs a member nothing: their meetings still work, they just appear by their Meet name or as SPEAKER_N.": (
        "Cada persona graba su propia muestra tras un aviso de consentimiento; puedes invitar, cancelar o eliminar, pero nunca subir audio en nombre de otra persona. Rechazar no le cuesta nada a un miembro: sus reuniones siguen funcionando, simplemente aparece con su nombre de Meet o como SPEAKER_N.",
        "Cada persona grava la seva pròpia mostra després d'un avís de consentiment; pots convidar, cancel·lar o eliminar, però mai pujar àudio en nom d'una altra persona. Rebutjar-ho no li costa res a un membre: les seves reunions continuen funcionant, simplement apareix amb el seu nom de Meet o com a SPEAKER_N.",
    ),
    "{n}s of audio": ("{n} s de audio", "{n} s d'àudio"),
    "consent recorded": ("consentimiento registrado", "consentiment registrat"),
    "Invited {date} · waiting for them to record": (
        "Invitado el {date} · pendiente de que grabe",
        "Convidat el {date} · pendent que gravi",
    ),
    "Declined the invitation": ("Rechazó la invitación", "Va rebutjar la invitació"),
    "Not enrolled": ("Sin registrar", "Sense registrar"),
    "Unlink {name} from {email}? The print stays; Aplec stops sharing meetings with this account automatically.": (
        "¿Desvincular a {name} de {email}? La huella se mantiene; Aplec deja de compartir reuniones con esta cuenta automáticamente.",
        "Vols desvincular {name} de {email}? L'empremta es manté; Aplec deixa de compartir reunions amb aquest compte automàticament.",
    ),
    "Unlink": ("Desvincular", "Desvincula"),
    "Delete the voiceprint for {name}? They can re-enroll anytime.": (
        "¿Eliminar la huella de voz de {name}? Puede volver a registrarla cuando quiera.",
        "Vols eliminar l'empremta de veu de {name}? La pot tornar a registrar quan vulgui.",
    ),
    "Delete the voiceprint for {name}?": ("¿Eliminar la huella de voz de {name}?", "Vols eliminar l'empremta de veu de {name}?"),
    "Cancel invite": ("Cancelar invitación", "Cancel·la la invitació"),
    "Invite to enroll": ("Invitar a registrarse", "Convida a registrar-se"),
    "Prints not linked to an account": ("Huellas sin vincular a una cuenta", "Empremtes sense vincular a cap compte"),
    "Enrolled before the consent flow existed (CLI), or the account left the allowlist. They still name speakers, but Aplec cannot share a meeting with that person automatically until the print is linked to their account. Link it here, or the member can re-enroll from Settings.": (
        "Registradas antes de que existiera el flujo de consentimiento (CLI), o la cuenta ya no está en la lista de usuarios permitidos. Siguen poniendo nombre a los hablantes, pero Aplec no puede compartir automáticamente una reunión con esa persona hasta que la huella esté vinculada a su cuenta. Vincúlala aquí, o el miembro puede volver a registrarla desde Ajustes.",
        "Registrades abans que existís el flux de consentiment (CLI), o el compte ja no és a la llista d'usuaris permesos. Encara posen nom als parlants, però Aplec no pot compartir automàticament una reunió amb aquesta persona fins que l'empremta estigui vinculada al seu compte. Vincula-la aquí, o el membre la pot tornar a registrar des de Configuració.",
    ),
    "source: {source}": ("origen: {source}", "origen: {source}"),
    "linked to {email}, no longer on the allowlist": (
        "vinculada a {email}, que ya no está en la lista de usuarios permitidos",
        "vinculada a {email}, que ja no és a la llista d'usuaris permesos",
    ),
    "Link to account…": ("Vincular a una cuenta…", "Vincula a un compte…"),
    "Link": ("Vincular", "Vincula"),
    "Worker unreachable; showing invitations only.": (
        "No se puede contactar con el worker; solo se muestran las invitaciones.",
        "No es pot contactar amb el worker; només es mostren les invitacions.",
    ),
    "Link failed": ("No se ha podido vincular", "No s'ha pogut vincular"),
    "Delete failed": ("No se ha podido eliminar", "No s'ha pogut eliminar"),

    # --- language names (tooltips on the language badges, summary languages) -----
    "English": ("Inglés", "Anglès"),
    "Chinese": ("Chino", "Xinès"),
    "Cantonese": ("Cantonés", "Cantonès"),
    "Hindi": ("Hindi", "Hindi"),
    "Spanish": ("Español", "Castellà"),
    "French": ("Francés", "Francès"),
    "Arabic": ("Árabe", "Àrab"),
    "Bengali": ("Bengalí", "Bengalí"),
    "Portuguese": ("Portugués", "Portuguès"),
    "Brazilian Portuguese": ("Portugués de Brasil", "Portuguès del Brasil"),
    "Russian": ("Ruso", "Rus"),
    "Urdu": ("Urdu", "Urdú"),
    "Indonesian": ("Indonesio", "Indonesi"),
    "German": ("Alemán", "Alemany"),
    "Japanese": ("Japonés", "Japonès"),
    "Turkish": ("Turco", "Turc"),
    "Tamil": ("Tamil", "Tàmil"),
    "Telugu": ("Telugu", "Telugu"),
    "Marathi": ("Maratí", "Marathi"),
    "Punjabi": ("Panyabí", "Panjabi"),
    "Gujarati": ("Guyaratí", "Gujarati"),
    "Kannada": ("Canarés", "Kannada"),
    "Malayalam": ("Malayalam", "Malaiàlam"),
    "Vietnamese": ("Vietnamita", "Vietnamita"),
    "Korean": ("Coreano", "Coreà"),
    "Persian": ("Persa", "Persa"),
    "Italian": ("Italiano", "Italià"),
    "Thai": ("Tailandés", "Tai"),
    "Polish": ("Polaco", "Polonès"),
    "Ukrainian": ("Ucraniano", "Ucraïnès"),
    "Dutch": ("Neerlandés", "Neerlandès"),
    "Greek": ("Griego", "Grec"),
    "Czech": ("Checo", "Txec"),
    "Swedish": ("Sueco", "Suec"),
    "Hebrew": ("Hebreo", "Hebreu"),
    "Danish": ("Danés", "Danès"),
    "Finnish": ("Finés", "Finès"),
    "Norwegian": ("Noruego", "Noruec"),
    "Norwegian Nynorsk": ("Noruego nynorsk", "Noruec nynorsk"),
    "Romanian": ("Rumano", "Romanès"),
    "Hungarian": ("Húngaro", "Hongarès"),
    "Slovak": ("Eslovaco", "Eslovac"),
    "Bulgarian": ("Búlgaro", "Búlgar"),
    "Croatian": ("Croata", "Croat"),
    "Serbian": ("Serbio", "Serbi"),
    "Malay": ("Malayo", "Malai"),
    "Tagalog": ("Tagalo", "Tagal"),
    "Swahili": ("Suajili", "Suahili"),
    "Hausa": ("Hausa", "Haussa"),
    "Yoruba": ("Yoruba", "Ioruba"),
    "Amharic": ("Amárico", "Amhàric"),
    "Burmese": ("Birmano", "Birmà"),
    "Khmer": ("Jemer", "Khmer"),
    "Nepali": ("Nepalí", "Nepalès"),
    "Sinhala": ("Cingalés", "Singalès"),
    "Azerbaijani": ("Azerbaiyano", "Azerbaidjanès"),
    "Kazakh": ("Kazajo", "Kazakh"),
    "Uzbek": ("Uzbeko", "Uzbek"),
    "Catalan": ("Catalán", "Català"),
}
