# backend/rag_docs_generation.py

class RAGDocsGenerator:
    def generate_rag_documents(self, log_data):
        """Generate RAG documents based on actual available signals"""
        documents = []
        
        # 1. FLIGHT OVERVIEW (Essential - covers 20% of queries)
        documents.append(self.create_flight_overview(log_data))
        
        # 2. ATTITUDE & CONTROL (Critical - covers 30% of queries)
        documents.append(self.create_attitude_control_document(log_data))
        
        # 3. GPS & NAVIGATION (Critical - covers 25% of queries)
        documents.append(self.create_gps_navigation_document(log_data))
        
        # 4. EKF QUATERNION DATA (Important - covers 15% of queries)
        documents.append(self.create_ekf_quaternion_document(log_data))
        
        # 5. PARAMETERS & CONFIGURATION (Important - covers 10% of queries)
        documents.append(self.create_parameters_document(log_data))
        
        # 6. ARDUPILOT REFERENCE (High priority - covers technical definitions)
        documents.append(self.create_ardupilot_reference_document(log_data))
        
        return documents
    def create_flight_overview(self, log_data):
        """High-level flight summary combining all available data sources"""
        duration = log_data.get('flight_duration_ms', 0) / 1000
        modes = log_data.get('flight_summary', {}).get('modes', [])
        events = log_data.get('flight_summary', {}).get('events', [])
        text_messages = log_data.get('flight_summary', {}).get('text_messages', [])
        vehicle = log_data.get('vehicle', 'unknown')
        filename = log_data.get('filename', 'unknown')
        
        # Count message types and samples
        msg_types = list(log_data.get('time_series_data', {}).keys())
        total_samples = sum(msg_data.get('sample_count', 0) 
                        for msg_data in log_data.get('time_series_data', {}).values())
        
        # Calculate actual flight duration from time series data
        actual_duration = 0
        if msg_types:
            time_ranges = []
            for msg_data in log_data.get('time_series_data', {}).values():
                time_range = msg_data.get('time_range', {})
                if time_range.get('start') and time_range.get('end'):
                    time_ranges.append((time_range['start'], time_range['end']))
            
            if time_ranges:
                start_time = min(start for start, end in time_ranges)
                end_time = max(end for start, end in time_ranges)
                actual_duration = (end_time - start_time) / 1000
        
        # Determine flight phase and characteristics
        flight_phase = "Unknown"
        flight_characteristics = []
        
        if modes:
            initial_mode = modes[0][1] if modes else "Unknown"
            final_mode = modes[-1][1] if modes else "Unknown"
            
            if initial_mode in ['LAND', 'RTL']:
                flight_phase = "Landing/Return"
            elif initial_mode in ['TAKEOFF', 'AUTO']:
                flight_phase = "Takeoff/Autonomous"
            elif initial_mode in ['LOITER', 'GUIDED']:
                flight_phase = "Active Flight"
            elif initial_mode in ['STABILIZE', 'ACRO']:
                flight_phase = "Manual Control"
            
            # Analyze mode changes
            if len(modes) > 1:
                flight_characteristics.append(f"Mode transitions: {initial_mode} → {final_mode}")
                mode_durations = []
                for i in range(len(modes) - 1):
                    duration = (modes[i+1][0] - modes[i][0]) / 1000
                    mode_durations.append(f"{modes[i][1]} ({duration:.1f}s)")
                flight_characteristics.append(f"Mode sequence: {' → '.join(mode_durations)}")
            else:
                flight_characteristics.append(f"Single mode flight: {initial_mode}")
        
        # Analyze data quality and coverage
        data_quality = []
        if total_samples > 0:
            data_quality.append(f"High data density: {total_samples:,} samples")
            
            # Check for data gaps
            sample_rates = []
            for msg_type, msg_data in log_data.get('time_series_data', {}).items():
                sample_count = msg_data.get('sample_count', 0)
                time_range = msg_data.get('time_range', {})
                if time_range.get('start') and time_range.get('end'):
                    duration = (time_range['end'] - time_range['start']) / 1000
                    if duration > 0:
                        sample_rate = sample_count / duration
                        sample_rates.append(f"{msg_type}: {sample_rate:.1f}Hz")
            
            if sample_rates:
                data_quality.append(f"Sample rates: {', '.join(sample_rates[:3])}{'...' if len(sample_rates) > 3 else ''}")
        
        # Analyze system health indicators
        health_indicators = []
        
        # Check for errors/warnings in text messages
        if text_messages:
            error_count = sum(1 for msg in text_messages if len(msg) >= 3 and 'error' in msg[2].lower())
            warning_count = sum(1 for msg in text_messages if len(msg) >= 3 and 'warning' in msg[2].lower())
            
            if error_count > 0:
                health_indicators.append(f"⚠️ {error_count} error messages")
            if warning_count > 0:
                health_indicators.append(f"⚠️ {warning_count} warning messages")
            if error_count == 0 and warning_count == 0:
                health_indicators.append("✅ No errors or warnings detected")
        
        # Check for events
        if events:
            health_indicators.append(f"{len(events)} flight events recorded")
        
        # Analyze specific data sources
        data_sources = []
        att_data = log_data.get('time_series_data', {}).get('ATT', {})
        gps_data = log_data.get('time_series_data', {}).get('GPS[0]', {})
        xkq_data = any(f'XKQ[{i}]' in log_data.get('time_series_data', {}) for i in range(3))
        
        if att_data:
            att_samples = att_data.get('sample_count', 0)
            data_sources.append(f"Attitude: {att_samples:,} samples")
        
        if gps_data:
            gps_samples = gps_data.get('sample_count', 0)
            data_sources.append(f"GPS: {gps_samples:,} samples")
        
        if xkq_data:
            data_sources.append("EKF Quaternions: Available")
        
        # Build comprehensive content
        content = f"""Flight Overview: {filename}
    Vehicle: {vehicle}
    Duration: {actual_duration:.1f} seconds ({actual_duration/60:.1f} minutes)
    Flight Phase: {flight_phase}

    Data Sources: {len(msg_types)} message types
    - {', '.join(msg_types[:5])}{'...' if len(msg_types) > 5 else ''}
    - Total Data Points: {total_samples:,}

    Flight Characteristics:
    {chr(10).join(f"- {char}" for char in flight_characteristics) if flight_characteristics else "- No mode data available"}

    Data Quality:
    {chr(10).join(f"- {quality}" for quality in data_quality) if data_quality else "- No data quality metrics"}

    System Health:
    {chr(10).join(f"- {indicator}" for indicator in health_indicators) if health_indicators else "- No health data available"}

    Available Data Sources:
    {chr(10).join(f"- {source}" for source in data_sources) if data_sources else "- No detailed data sources"}

    This flight log contains comprehensive telemetry data from a {vehicle} flight lasting {actual_duration:.1f} seconds.
    The log includes {len(msg_types)} different data streams with over {total_samples:,} individual measurements."""
        
        # Add detailed mode changes if available
        if modes:
            content += f"\n\nDetailed Mode Changes:\n"
            for i, (timestamp, mode) in enumerate(modes):
                duration_str = ""
                if i < len(modes) - 1:
                    next_timestamp = modes[i + 1][0]
                    duration = (next_timestamp - timestamp) / 1000
                    duration_str = f" (for {duration:.1f}s)"
                content += f"- {timestamp/1000:.1f}s: {mode}{duration_str}\n"
        
        # Add recent important messages if available
        if text_messages:
            important_messages = [msg for msg in text_messages if len(msg) >= 3 and 
                                any(keyword in msg[2].lower() for keyword in 
                                ['error', 'warning', 'fail', 'fault', 'gps', 'ekf', 'armed', 'disarmed', 'land'])]
            if important_messages:
                content += f"\nRecent Important Messages:\n"
                for msg in important_messages[-3:]:  # Last 3 important messages
                    content += f"- {msg[0]/1000:.1f}s: {msg[2]}\n"
        
        return {
            "document_id": f"{log_data['log_id']}_overview",
            "document_type": "flight_overview",
            "title": f"Flight Overview - {filename}",
            "content": content.strip(),
            "metadata": {
                "filename": filename,
                "vehicle": vehicle,
                "duration_seconds": actual_duration,
                "message_types": msg_types,
                "total_samples": total_samples,
                "mode_changes": len(modes),
                "events": len(events),
                "text_messages": len(text_messages),
                "flight_phase": flight_phase,
                "has_attitude": bool(att_data),
                "has_gps": bool(gps_data),
                "has_ekf": xkq_data
            },
            "searchable_fields": ["vehicle", "filename", "duration", "message_types", "flight_phase", "modes", "health"]
    }
    def create_attitude_control_document(self, log_data):
        """Attitude control data - most critical for flight stability debugging"""
        content = "Attitude Control and Flight Stability:\n\n"
        
        att_data = log_data.get('time_series_data', {}).get('ATT', {})
        if att_data:
            data = att_data.get('data', {})
            sample_count = att_data.get('sample_count', 0)
            time_range = att_data.get('time_range', {})
            duration = (time_range.get('end', 0) - time_range.get('start', 0)) / 1000
            
            content += f"Attitude Data (ATT):\n"
            content += f"- Samples: {sample_count:,} over {duration:.1f}s\n"
            content += f"- Sample Rate: {sample_count/duration:.1f} Hz\n"
            content += f"- Fields: {', '.join(att_data.get('fields', []))}\n"
            
            # Calculate attitude statistics
            rolls = data.get('Roll', [])
            pitches = data.get('Pitch', [])
            yaws = data.get('Yaw', [])
            des_rolls = data.get('DesRoll', [])
            des_pitches = data.get('DesPitch', [])
            des_yaws = data.get('DesYaw', [])
            err_rp = data.get('ErrRP', [])
            err_yaw = data.get('ErrYaw', [])

            if rolls and pitches and yaws:
                content += f"\nAttitude Ranges:\n"
                content += f"- Roll: {min(rolls):.1f}° to {max(rolls):.1f}°\n"
                content += f"- Pitch: {min(pitches):.1f}° to {max(pitches):.1f}°\n"
                content += f"- Yaw: {min(yaws):.1f}° to {max(yaws):.1f}°\n"
                
                # Calculate attitude errors if desired values are available
                if des_rolls and des_pitches and des_yaws:
                    roll_errors = [abs(r - dr) for r, dr in zip(rolls, des_rolls)]
                    pitch_errors = [abs(p - dp) for p, dp in zip(pitches, des_pitches)]
                    yaw_errors = [abs(y - dy) for y, dy in zip(yaws, des_yaws)]
                    
                    content += f"\nAttitude Control Performance:\n"
                    content += f"- Roll Error: {min(roll_errors):.1f}° to {max(roll_errors):.1f}° (avg: {sum(roll_errors)/len(roll_errors):.1f}°)\n"
                    content += f"- Pitch Error: {min(pitch_errors):.1f}° to {max(pitch_errors):.1f}° (avg: {sum(pitch_errors)/len(pitch_errors):.1f}°)\n"
                    content += f"- Yaw Error: {min(yaw_errors):.1f}° to {max(yaw_errors):.1f}° (avg: {sum(yaw_errors)/len(yaw_errors):.1f}°)\n"
            
            # Control error analysis
            if err_rp:
                content += f"\nControl System Errors:\n"
                content += f"- Roll/Pitch Error: {min(err_rp):.1f} to {max(err_rp):.1f} (avg: {sum(err_rp)/len(err_rp):.1f})\n"
            if err_yaw:
                content += f"- Yaw Error: {min(err_yaw):.1f} to {max(err_yaw):.1f} (avg: {sum(err_yaw)/len(err_yaw):.1f})\n"
            
            # AEKF analysis (EKF health indicator)
            aekf = data.get('AEKF', [])
            if aekf:
                content += f"\nEKF Health (AEKF):\n"
                content += f"- Range: {min(aekf):.1f} to {max(aekf):.1f}\n"
                content += f"- Average: {sum(aekf)/len(aekf):.1f}\n"
        else:
            content += "No attitude data available\n"
        
        return {
            "document_id": f"{log_data['log_id']}_attitude_control",
            "document_type": "attitude_control",
            "title": "Attitude Control and Flight Stability",
            "content": content.strip(),
            "metadata": {
                "has_attitude": bool(att_data),
                "attitude_samples": att_data.get('sample_count', 0) if att_data else 0
            },
            "searchable_fields": ["attitude", "control", "stability", "roll", "pitch", "yaw", "errors", "ekf"],
        }
    
    def create_gps_navigation_document(self, log_data):
        """GPS navigation and positioning data"""
        content = "GPS Navigation and Positioning:\n\n"
        
        gps_data = log_data.get('time_series_data', {}).get('GPS[0]', {})
        if gps_data:
            data = gps_data.get('data', {})
            sample_count = gps_data.get('sample_count', 0)
            time_range = gps_data.get('time_range', {})
            duration = (time_range.get('end', 0) - time_range.get('start', 0)) / 1000
            
            content += f"GPS Data (GPS[0]):\n"
            content += f"- Samples: {sample_count:,} over {duration:.1f}s\n"
            content += f"- Sample Rate: {sample_count/duration:.1f} Hz\n"
            content += f"- Fields: {', '.join(gps_data.get('fields', []))}\n"
            
            # Calculate GPS health metrics
            lats = data.get('Lat', [])
            lngs = data.get('Lng', [])
            alts = data.get('Alt', [])
            sats = data.get('NSats', [])
            hdop = data.get('HDop', [])
            speeds = data.get('Spd', [])
            headings = data.get('GCrs', [])
            vz = data.get('VZ', [])
            
            if lats and lngs:
                content += f"\nPosition Data:\n"
                content += f"- Latitude Range: {min(lats)/1e7:.6f}° to {max(lats)/1e7:.6f}°\n"
                content += f"- Longitude Range: {min(lngs)/1e7:.6f}° to {max(lngs)/1e7:.6f}°\n"
                
                # Calculate distance traveled
                if len(lats) > 1:
                    import math
                    distances = []
                    for i in range(1, len(lats)):
                        lat1, lng1 = lats[i-1]/1e7, lngs[i-1]/1e7
                        lat2, lng2 = lats[i]/1e7, lngs[i]/1e7
                        # Simple distance calculation
                        dist = math.sqrt((lat2-lat1)**2 + (lng2-lng1)**2) * 111000  # rough meters
                        distances.append(dist)
                    total_distance = sum(distances)
                    content += f"- Total Distance: {total_distance:.1f}m\n"
            
            if alts:
                content += f"- Altitude Range: {min(alts)/1000:.1f}m to {max(alts)/1000:.1f}m\n"
            
            if sats:
                content += f"\nGPS Health:\n"
                content += f"- Satellite Count: {min(sats)} to {max(sats)} satellites\n"
                content += f"- Average Satellites: {sum(sats)/len(sats):.1f}\n"
            
            if hdop:
                content += f"- HDOP Range: {min(hdop):.1f} to {max(hdop):.1f} (lower is better)\n"
                content += f"- Average HDOP: {sum(hdop)/len(hdop):.1f}\n"
            
            if speeds:
                content += f"\nVelocity Data:\n"
                content += f"- Ground Speed: {min(speeds):.1f} to {max(speeds):.1f} m/s\n"
                content += f"- Average Speed: {sum(speeds)/len(speeds):.1f} m/s\n"
            
            if vz:
                content += f"- Vertical Speed: {min(vz):.1f} to {max(vz):.1f} m/s\n"
        else:
            content += "No GPS data available\n"
        
        return {
            "document_id": f"{log_data['log_id']}_gps_navigation",
            "document_type": "gps_navigation",
            "title": "GPS Navigation and Positioning",
            "content": content.strip(),
            "metadata": {
                "has_gps": bool(gps_data),
                "gps_samples": gps_data.get('sample_count', 0) if gps_data else 0
            },
            "searchable_fields": ["gps", "navigation", "position", "satellites", "hdop", "speed"]
        }
    
    def create_ekf_quaternion_document(self, log_data):
        """EKF quaternion data for attitude estimation"""
        content = "EKF Quaternion Attitude Estimation:\n\n"
        
        # Check all XKQ instances
        xkq_data = {}
        for i in range(3):  # XKQ[0], XKQ[1], XKQ[2]
            msg_type = f'XKQ[{i}]'
            if msg_type in log_data.get('time_series_data', {}):
                xkq_data[msg_type] = log_data['time_series_data'][msg_type]
        
        if xkq_data:
            content += f"EKF Quaternion Data Available:\n"
            for msg_type, data in xkq_data.items():
                sample_count = data.get('sample_count', 0)
                time_range = data.get('time_range', {})
                duration = (time_range.get('end', 0) - time_range.get('start', 0)) / 1000
                content += f"- {msg_type}: {sample_count:,} samples over {duration:.1f}s\n"
            
            # Analyze quaternion data from primary instance (XKQ[0])
            primary_data = xkq_data.get('XKQ[0]', {})
            if primary_data:
                data = primary_data.get('data', {})
                q1 = data.get('Q1', [])
                q2 = data.get('Q2', [])
                q3 = data.get('Q3', [])
                q4 = data.get('Q4', [])
                
                if q1 and q2 and q3 and q4:
                    content += f"\nQuaternion Analysis (XKQ[0]):\n"
                    content += f"- Q1 Range: {min(q1):.3f} to {max(q1):.3f}\n"
                    content += f"- Q2 Range: {min(q2):.3f} to {max(q2):.3f}\n"
                    content += f"- Q3 Range: {min(q3):.3f} to {max(q3):.3f}\n"
                    content += f"- Q4 Range: {min(q4):.3f} to {max(q4):.3f}\n"
                    
                    # Check quaternion normalization
                    import math
                    norms = [math.sqrt(q1[i]**2 + q2[i]**2 + q3[i]**2 + q4[i]**2) for i in range(len(q1))]
                    if norms:
                        content += f"- Quaternion Norm: {min(norms):.3f} to {max(norms):.3f} (should be ~1.0)\n"
                        avg_norm = sum(norms)/len(norms)
                        content += f"- Average Norm: {avg_norm:.3f}\n"
                        
                        if avg_norm < 0.95 or avg_norm > 1.05:
                            content += f"- WARNING: Quaternion normalization issues detected!\n"
        else:
            content += "No EKF quaternion data available\n"
        
        return {
            "document_id": f"{log_data['log_id']}_ekf_quaternion",
            "document_type": "ekf_quaternion",
            "title": "EKF Quaternion Attitude Estimation",
            "content": content.strip(),
            "metadata": {
                "has_quaternion": bool(xkq_data),
                "quaternion_instances": len(xkq_data)
            },
            "searchable_fields": ["ekf", "quaternion", "attitude", "estimation", "q1", "q2", "q3", "q4"]
        }
    
    def create_parameters_document(self, log_data):
        """Configuration parameters and settings"""
        content = "Configuration Parameters:\n\n"
        
        parameters = log_data.get('parameters', {})
        if parameters:
            # Get parameter change array
            change_array = parameters.get('changeArray', [])
            if change_array:
                content += f"Parameter Changes: {len(change_array)} changes recorded\n\n"
                
                # Group parameters by category
                categories = {
                    'System': [p for p in change_array if any(x in p[1] for x in ['SYSID', 'FORMAT', 'MAV_TYPE'])],
                    'Frame': [p for p in change_array if 'FRAME' in p[1]],
                    'Navigation': [p for p in change_array if any(x in p[1] for x in ['WPNAV', 'RTL', 'FENCE', 'GPS'])],
                    'Sensors': [p for p in change_array if any(x in p[1] for x in ['EK2', 'EK3', 'GPS', 'MAG', 'BARO'])],
                    'Control': [p for p in change_array if any(x in p[1] for x in ['RATE', 'STAB', 'ACRO', 'PID'])],
                    'Safety': [p for p in change_array if any(x in p[1] for x in ['ARM', 'DISARM', 'BATT', 'FENCE'])],
                }
                
                for category, params in categories.items():
                    if params:
                        content += f"{category} Parameters:\n"
                        for param in params[:10]:  # Limit to 10 per category
                            if len(param) >= 3:
                                timestamp = param[0]/1000
                                name = param[1]
                                value = param[2]
                                content += f"- {timestamp:.1f}s: {name} = {value}\n"
                        if len(params) > 10:
                            content += f"... and {len(params) - 10} more\n"
                        content += "\n"
            else:
                content += "No parameter changes recorded\n"
        else:
            content += "No parameter data available\n"
        
        return {
            "document_id": f"{log_data['log_id']}_parameters",
            "document_type": "parameters",
            "title": "Configuration Parameters",
            "content": content.strip(),
            "metadata": {
                "parameter_changes": len(parameters.get('changeArray', [])) if parameters else 0
            },
            "searchable_fields": ["parameters", "configuration", "settings", "changes"]
        }
    
    def create_system_health_document(self, log_data):
        """Overall system health and diagnostics"""
        content = "System Health and Diagnostics:\n\n"
        
        # Flight summary data
        flight_summary = log_data.get('flight_summary', {})
        modes = flight_summary.get('modes', [])
        events = flight_summary.get('events', [])
        text_messages = flight_summary.get('text_messages', [])
        
        content += f"Flight Summary:\n"
        content += f"- Mode Changes: {len(modes)}\n"
        content += f"- Events: {len(events)}\n"
        content += f"- Text Messages: {len(text_messages)}\n"
        
        if modes:
            content += f"\nMode Changes:\n"
            for timestamp, mode in modes:
                content += f"- {timestamp/1000:.1f}s: {mode}\n"
        
        if events:
            content += f"\nFlight Events:\n"
            for event in events[-5:]:  # Last 5 events
                if len(event) >= 2:
                    content += f"- {event[0]/1000:.1f}s: {event[1]}\n"
        
        # Check for important messages
        if text_messages:
            important_messages = [msg for msg in text_messages if len(msg) >= 3 and 
                                any(keyword in msg[2].lower() for keyword in 
                                ['error', 'warning', 'fail', 'fault', 'gps', 'ekf', 'armed', 'disarmed', 'bad'])]
            if important_messages:
                content += f"\nImportant Messages:\n"
                for msg in important_messages:  # Last 5 important messages
                    content += f"- {msg[0]/1000:.1f}s: {msg[2]}\n"
        
        return {
            "document_id": f"{log_data['log_id']}_system_health",
            "document_type": "system_health",
            "title": "System Health and Diagnostics",
            "content": content.strip(),
            "metadata": {
                "mode_changes": len(modes),
                "events": len(events),
                "messages": len(text_messages)
            },
            "searchable_fields": ["health", "diagnostics", "modes", "events", "messages", "errors"]
        }
    def create_ardupilot_reference_document(self, log_data):
        """Create reference document with ArduPilot message type definitions"""
        try:
            # Load the ArduPilot documentation index
            import json
            import os
            
            docs_path = 'static/ardupilot_index.json'
            if not os.path.exists(docs_path):
                return {
                    "title": "ArduPilot Message Reference",
                    "content": "ArduPilot documentation not available. Please run docs_parser.py to generate reference data.",
                    "document_type": "reference",
                    "priority": "low"
                }
            
            with open(docs_path, 'r') as f:
                docs_data = json.load(f)
            
            # Find the log messages document
            log_docs = [doc for doc in docs_data.get('docs', []) if doc.get('type') == 'ardupilot_log_messages']
            if not log_docs:
                return {
                    "title": "ArduPilot Message Reference", 
                    "content": "No ArduPilot log message documentation found.",
                    "document_type": "reference",
                    "priority": "low"
                }
            
            log_doc = log_docs[0]
            message_sections = log_doc.get('message_sections', [])
            
            # Filter for relevant message types
            relevant_types = ['ATT', 'GPS', 'XKQ', 'PARM']
            relevant_sections = [section for section in message_sections 
                            if section.get('message_type') in relevant_types]
            
            if not relevant_sections:
                return {
                    "title": "ArduPilot Message Reference",
                    "content": "No relevant message types found in documentation.",
                    "document_type": "reference", 
                    "priority": "low"
                }
            
            # Build reference content
            content_parts = [
                "# ArduPilot Message Type Reference",
                "",
                "This document provides technical definitions for the message types present in this log file.",
                "Use this reference to understand field meanings, units, and engineering significance.",
                ""
            ]
            
            for section in relevant_sections:
                msg_type = section.get('message_type', 'Unknown')
                description = section.get('description', 'No description available')
                table_data = section.get('table_data', [])
                
                content_parts.extend([
                    f"## {msg_type} - {description}",
                    ""
                ])
                
                if table_data:
                    content_parts.append("### Field Definitions:")
                    content_parts.append("")
                    
                    for item in table_data:
                        content_parts.append(str(item))
                    content_parts.append("") # Add a blank line for separation
                else:
                    content_parts.append("*No field definitions available*")
                    content_parts.append("")
            
            # Add engineering context
            content_parts.extend([
                "## Engineering Context",
                "",
                "### ATT (Attitude) Message:",
                "- Contains vehicle orientation data (roll, pitch, yaw)",
                "- Critical for flight control and stability analysis", 
                "- DesRoll/DesPitch/DesYaw are commanded values",
                "- Roll/Pitch/Yaw are actual measured values",
                "",
                "### GPS Message:",
                "- Contains Global Navigation Satellite System data",
                "- Essential for position, velocity, and navigation analysis",
                "- Status field indicates GPS fix quality (0=no fix, 2=2D fix, 3=3D fix)",
                "- HDop/VDop indicate position accuracy dilution",
                "",
                "### XKQ (EKF Quaternion) Message:",
                "- Contains Extended Kalman Filter quaternion data",
                "- Represents rotation from NED (North-East-Down) to XYZ (autopilot) axes",
                "- Critical for understanding sensor fusion and attitude estimation",
                "- Q1, Q2, Q3, Q4 are quaternion components",
                "",
                "### PARM (Parameters) Message:",
                "- Contains vehicle configuration parameters",
                "- Essential for understanding vehicle setup and tuning",
                "- Parameters control flight behavior, limits, and safety features",
                "- Name field contains parameter name, Value contains current setting",
                ""
            ])
            
            return {
                "title": "ArduPilot Message Reference",
                "content": "\n".join(content_parts),
                "document_type": "reference",
                "priority": "high",
                "message_types_covered": [s.get('message_type') for s in relevant_sections]
            }
            
        except Exception as e:
            return {
                "title": "ArduPilot Message Reference",
                "content": f"Error loading ArduPilot documentation: {str(e)}",
                "document_type": "reference",
                "priority": "low"
            }