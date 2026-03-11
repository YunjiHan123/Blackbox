from events.event_types import EVENT_ALERT_TITLES, EVENT_LABELS


def handle_event(
    frame,
    event_type,
    logger,
    saver,
    alert_state,
    ensure_alert_window,
    subtitle=None,
    log_message=None,
    save_prefix=None,
):

    logger.log(log_message or f"{event_type} detected")
    saver.save(frame, save_prefix or event_type)
    alert_state.activate(
        frame,
        EVENT_ALERT_TITLES[event_type],
        subtitle or EVENT_LABELS[event_type],
    )
    ensure_alert_window()
