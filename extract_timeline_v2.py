#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract clean timeline from Claude JSONL session files for 2026-02-07."""

import json
import sys
from datetime import datetime
from pathlib import Path

def parse_timestamp(ts_str):
    """Parse ISO timestamp to datetime."""
    try:
        return datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
    except:
        return None

def extract_text_from_content(content):
    """Extract text from content array."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict):
                if item.get('type') == 'text':
                    texts.append(item.get('text', ''))
            elif isinstance(item, str):
                texts.append(item)
        return ' '.join(texts).strip()
    return str(content)

def parse_session_file(filepath, session_name, approx_time):
    """Parse a single JSONL file and extract events."""
    events = []

    print(f"Processing {filepath.name}...")

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Extract timestamp
            timestamp = None
            if 'timestamp' in obj:
                timestamp = parse_timestamp(obj['timestamp'])
            elif 'message' in obj and isinstance(obj['message'], dict):
                if 'timestamp' in obj['message']:
                    timestamp = parse_timestamp(obj['message']['timestamp'])

            if not timestamp:
                continue

            # Only process events from 2026-02-07
            if timestamp.date() != datetime(2026, 2, 7).date():
                continue

            message = obj.get('message', {})
            if not isinstance(message, dict):
                continue

            role = message.get('role', '')
            content = message.get('content', [])

            # User messages
            if role == 'user':
                text = extract_text_from_content(content)
                # Filter out system notifications and empty messages
                if text and not text.startswith('<task-notification>') and not text.startswith('[Request interrupted'):
                    # Truncate very long messages
                    if len(text) > 300:
                        text = text[:300] + '...'
                    events.append({
                        'time': timestamp,
                        'type': 'user',
                        'text': text
                    })

            # Assistant tool use
            elif role == 'assistant':
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get('type') == 'tool_use':
                            tool_name = item.get('name', 'unknown')
                            tool_input = item.get('input', {})

                            # Track important file operations
                            if tool_name == 'Write':
                                file_path = tool_input.get('file_path', '')
                                if file_path:
                                    filename = Path(file_path).name
                                    events.append({
                                        'time': timestamp,
                                        'type': 'file_create',
                                        'text': filename
                                    })

                            elif tool_name == 'Edit':
                                file_path = tool_input.get('file_path', '')
                                if file_path:
                                    filename = Path(file_path).name
                                    events.append({
                                        'time': timestamp,
                                        'type': 'file_edit',
                                        'text': filename
                                    })

                            elif tool_name == 'Bash':
                                cmd = tool_input.get('command', '')
                                # Only track Python executions
                                if 'python' in cmd.lower() and '.py' in cmd:
                                    # Extract script name
                                    for part in cmd.split():
                                        if '.py' in part:
                                            script = Path(part.strip("'\"")).name
                                            events.append({
                                                'time': timestamp,
                                                'type': 'run_script',
                                                'text': script
                                            })
                                            break

    print(f"  Found {len(events)} events")
    return events

def format_timeline_md(all_events):
    """Format events into markdown timeline."""
    if not all_events:
        return "No events found."

    # Sort by time
    all_events.sort(key=lambda x: x['time'])

    output = []
    output.append("# 2026-02-07 개발 일지 타임라인\n")

    current_hour = None

    for event in all_events:
        hour = event['time'].hour
        time_str = event['time'].strftime('%H:%M:%S')

        # Add hour separator
        if hour != current_hour:
            current_hour = hour
            output.append(f"\n## {hour:02d}시대\n")

        event_type = event['type']
        text = event['text']

        if event_type == 'user':
            output.append(f"- **{time_str}** 요청: {text}")
        elif event_type == 'file_create':
            output.append(f"  - {time_str} 📝 생성: `{text}`")
        elif event_type == 'file_edit':
            output.append(f"  - {time_str} ✏️ 수정: `{text}`")
        elif event_type == 'run_script':
            output.append(f"  - {time_str} ▶️ 실행: `{text}`")

    return '\n'.join(output)

def main():
    """Main entry point."""
    # Session files
    sessions = [
        ("C:\\Users\\lay4\\.claude\\projects\\D--task-hoban\\d07f876f-371e-421f-97d0-32943fd45ea3.jsonl",
         "d07f876f", "06:26"),
        ("C:\\Users\\lay4\\.claude\\projects\\D--task-hoban\\4ac77157-7d76-4af0-b0e8-be7b1f065c97.jsonl",
         "4ac77157", "11:48"),
        ("C:\\Users\\lay4\\.claude\\projects\\D--task-hoban\\676bf472-afde-42c7-8cae-384042afd6fe.jsonl",
         "676bf472", "18:49"),
        ("C:\\Users\\lay4\\.claude\\projects\\D--task-hoban\\fdc14112-cd8d-4b29-a35c-ded35730f8f0.jsonl",
         "fdc14112", "21:25"),
    ]

    all_events = []

    for filepath_str, session_id, approx_time_str in sessions:
        filepath = Path(filepath_str)

        if not filepath.exists():
            print(f"Skipping missing file: {filepath.name}")
            continue

        approx_time = datetime(2026, 2, 7,
                               int(approx_time_str.split(':')[0]),
                               int(approx_time_str.split(':')[1]))

        events = parse_session_file(filepath, session_id, approx_time)
        all_events.extend(events)

    # Generate markdown
    output_md = format_timeline_md(all_events)

    # Save to file
    output_file = Path('D:\\task\\hoban\\TIMELINE_2026-02-07.md')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output_md)

    print(f"\n✅ Timeline saved to: {output_file}")
    print(f"📊 Total events: {len(all_events)}")

if __name__ == '__main__':
    main()
