pdate(self):
        ret, self.frame = self.cap.read()
        if not ret:
            return False

        if self.mode == "collect":
            cv2.imshow('Bus Camera Test', self.frame)
        else:
            results = self.model(self.frame, conf=CONFIG.CONF_THRESHOLD)[0]
            annotated = results.plot()

            # List to collect (x_center, text) for this frame
            current_numbers = []

            if CONFIG.USE_OCR and self.reader:
                for idx, box in enumerate(results.boxes):
                    cls = int(box.cls[0])
                    if cls == 1:  # route_number class
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        # Calculate x_center for sorting
                        x_center = (x1 + x2) // 2
                        crop = self.frame[y1:y2, x1:x2]
                        timestamp = int(time.time())

                        if CONFIG.SAVE_CROP_IMAGES:
                            cv2.imwrite(f"debug_crop_{timestamp}_{idx}.jpg", crop)

                        if CONFIG.PREPROCESS_OCR:
                            processed_crop = self.preprocess_for_ocr(crop, timestamp, idx)
                            if processed_crop is None:
                                continue
                        else:
                            processed_crop = crop

                        ocr_result = self.reader.readtext(processed_crop)
                        print(f"OCR raw result: {ocr_result}")

                        if ocr_result and ocr_result[0][2] > CONFIG.OCR_CONF_THRESHOLD:
                            text = ocr_result[0][1]
                            confidence = ocr_result[0][2]
                            display_text = f"{text} ({confidence:.2f})"
                            color = (0, 255, 0)

                            # Add to current frame list
                            current_numbers.append((x_center, text))

                            # Log arrival (unchanged)
                            current_time = time.time()
                            if text not in self.bus_log:
                                self.bus_log[text] = current_time
                                print(f"New bus detected: route {text} at {time.ctime(current_time)}")
                            self.bus_last_seen[text] = current_time
                        else:
                            display_text = "?"
                            color = (0, 0, 255)
                            if ocr_result:
                                print(f"OCR confidence too low: {ocr_result[0][2]} < {CONFIG.OCR_CONF_THRESHOLD}")
                            else:
                                print("OCR returned no text")

                        cv2.putText(annotated, display_text, (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # After processing all detections, sort and print left-to-right order
            if current_numbers:
                # Sort by x_center (left to right)
                current_numbers.sort(key=lambda item: item[0])
                numbers_in_order = [text for _, text in current_numbers]
                print(" ".join(numbers_in_order))

            # Remove old buses (unchanged)
            now = time.time()
            to_remove = [route for route, last in self.bus_last_seen.items() if now - last > self.seen_timeout]
            for route in to_remove:
                del self.bus_log[route]
                del self.bus_last_seen[route]
                print(f"Route {route} left the stop")

            cv2.imshow('Bus Detection', annotated)

        # Key handling (unchanged) ...
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return False
        elif key == ord('m'):
            self.mode = "detect" if self.mode == "collect" else "collect"
            print(f"Switched to {self.mode} mode")
            if self.mode == "detect" and self.model is None:
                self.model = YOLO(CONFIG.MODEL_PATH)
                if CONFIG.USE_OCR:
                    self.reader = easyocr.Reader(['en'])
            return True
        elif key == ord('o'):
            if self.mode == "detect":
                if not self.bus_log:
                    print("No buses have been detected yet.")
                else:
                    sorted_buses = sorted(self.bus_log.items(), key=lambda x: x[1])
                    print("\n--- Bus Arrival Order ---")
                    for idx, (route, first_time) in enumerate(sorted_buses, 1):
                        print(f"{idx}. Route {route} at {time.ctime(first_time)}")
                    print("------------------------")
            else:
                print("Order only available in detect mode")
        elif self.mode == "collect" and key == 32:
            self.sampler.capturePhoto(self)

        return True