#!/usr/bin/env python3
"""Extract timeline from Claude JSONL session files."""

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
                elif item.get('type') == 'tool_use':
                    tool_name = item.get('name', 'unknown')
                    texts.append(f"[TOOL: {tool_name}]")
            elif isinstance(item, str):
                texts.append(item)
        return ' '.join(texts)
    return str(content)

def parse_session_file(filepath, session_name, approx_time):
    """Parse a single JSONL file and extract events."""
    events = []

    print(f"\n[PROCESSING] {filepath.name} ({filepath.stat().st_size / 1024:.0f} KB)")

    line_count = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line_count += 1
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

            # Check message type
            msg_type = obj.get('type', '')

            # Handle different message structures
            message = obj.get('message', {})
            if isinstance(message, dict):
                role = message.get('role', '')
                content = message.get('content', [])

                # User messages
                if role == 'user' or msg_type == 'human':
                    text = extract_text_from_content(content)
                    if text and not text.startswith('[TOOL:'):
                        events.append({
                            'time': timestamp or approx_time,
                            'type': 'user',
                            'text': text[:200]  # Truncate long messages
                        })

                # Assistant tool use
                elif role == 'assistant':
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get('type') == 'tool_use':
                                tool_name = item.get('name', 'unknown')
                                tool_input = item.get('input', {})

                                # Extract key info based on tool
                                if tool_name in ['Write', 'Edit']:
                                    file_path = tool_input.get('file_path', '')
                                    events.append({
                                        'time': timestamp or approx_time,
                                        'type': 'file_op',
                                        'text': f"{tool_name}: {Path(file_path).name}"
                                    })
                                elif tool_name == 'Bash':
                                    cmd = tool_input.get('command', '')
                                    if 'python' in cmd.lower():
                                        events.append({
                                            'time': timestamp or approx_time,
                                            'type': 'exec',
                                            'text': f"Execute: {cmd[:100]}"
                                        })

    print(f"  Processed {line_count} lines, found {len(events)} events")
    return events

def format_timeline(all_events):
    """Format events into timeline text."""
    if not all_events:
        return "No events found."

    # Sort by time
    all_events.sort(key=lambda x: x['time'])

    output = []
    for event in all_events:
        time_str = event['time'].strftime('%H:%M:%S')
        event_type = event['type']
        text = event['text']

        if event_type == 'user':
            output.append(f"  {time_str} 사용자: \"{text}\"")
        elif event_type == 'file_op':
            output.append(f"  {time_str} 파일: {text}")
        elif event_type == 'exec':
            output.append(f"  {time_str} 실행: {text}")

    return '\n'.join(output)

def main():
    """Main entry point."""
    print("[OBJECTIVE] Extract timeline from 4 Claude JSONL session files for 2026-02-07")
    print("[STAGE:begin:data_loading]")

    # Session files with approximate times
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

    results = []

    for filepath_str, session_id, approx_time_str in sessions:
        filepath = Path(filepath_str)

        if not filepath.exists():
            print(f"\n[WARNING] File not found: {filepath}")
            continue

        # Create approximate datetime
        approx_time = datetime(2026, 2, 7, int(approx_time_str.split(':')[0]),
                               int(approx_time_str.split(':')[1]))

        events = parse_session_file(filepath, session_id, approx_time)

        if events:
            results.append({
                'session_id': session_id,
                'approx_time': approx_time_str,
                'events': events
            })

    print("\n[STAGE:status:success]")
    print("[STAGE:end:data_loading]")

    # Format output
    print("\n" + "="*80)
    print("TIMELINE FOR 2026-02-07")
    print("="*80)

    for result in results:
        print(f"\n## 세션 {result['session_id'][:8]} (~{result['approx_time']})")
        print(format_timeline(result['events']))

    print(f"\n[FINDING] Extracted timeline from {len(results)} sessions")

if __name__ == '__main__':
    main()
