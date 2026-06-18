import streamlit.components.v1 as components


def render_animated_header():
    """
    Render the CADEMAS-ML pipeline animation in a top-to-bottom flow:
    four inputs → processing → Ri & Ci → prioritization (λ) → analysis views.
    """
    html_code = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Geist+Mono:wght@100..900&display=swap');
        </style>
        <style>
            body {
                margin: 0;
                padding: 0;
                background-color: transparent;
                font-family: 'Geist Mono', monospace;
                overflow: hidden;
            }
            .container {
                width: 100%;
                height: 455px;
                background: linear-gradient(180deg, #f8fafc 0%, #eef2f7 55%, #f8fafc 100%);
                border-radius: 10px;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            svg { width: 100%; height: 100%; max-width: 820px; }

            .node-rect {
                fill: #ffffff;
                stroke: #cbd5e1;
                stroke-width: 2px;
                rx: 6px;
            }
            .text-title {
                fill: #111827;
                font-size: 11px;
                font-weight: 600;
                font-family: 'Geist Mono', monospace;
                pointer-events: none;
            }
            .text-sub {
                fill: #64748b;
                font-size: 9px;
                font-family: 'Geist Mono', monospace;
                pointer-events: none;
            }
            .stroke-input { stroke: #94a3b8; }
            .stroke-ml { stroke: #2563eb; }
            .stroke-fuzzy { stroke: #ea580c; }
            .stroke-hybrid { stroke: #7c3aed; }
            .stroke-output { stroke: #059669; }
            .stroke-process { stroke: #475569; }

            .path-line {
                fill: none;
                stroke: #94a3b8;
                stroke-width: 2px;
                opacity: 0.6;
            }
            .dot { filter: drop-shadow(0 0 4px rgba(15,23,42,0.25)); }
            .dot-input { fill: #64748b; }
            .dot-ml { fill: #2563eb; }
            .dot-fuzzy { fill: #ea580c; }
            .dot-hybrid { fill: #7c3aed; }
            .dot-output { fill: #059669; }
        </style>
    </head>
    <body>
        <div class="container">
            <svg viewBox="0 0 530 440" preserveAspectRatio="xMidYMid meet">
                <defs>
                    <!-- Inputs → Processing -->
                    <path id="in1" d="M 70 62 L 70 108 L 265 108 L 265 128" />
                    <path id="in2" d="M 200 62 L 200 108 L 265 108 L 265 128" />
                    <path id="in3" d="M 330 62 L 330 108 L 265 108 L 265 128" />
                    <path id="in4" d="M 460 62 L 460 108 L 265 108 L 265 128" />
                    <!-- Processing → Ri / Ci -->
                    <path id="proc-ri" d="M 240 172 L 200 172 L 200 198" />
                    <path id="proc-ci" d="M 290 172 L 330 172 L 330 198" />
                    <!-- Ri / Ci → Prioritization -->
                    <path id="ri-prior" d="M 200 242 L 200 268 L 265 268 L 265 288" />
                    <path id="ci-prior" d="M 330 242 L 330 268 L 265 268 L 265 288" />
                    <!-- Prioritization → Outputs -->
                    <path id="prior-out1" d="M 230 332 L 230 358 L 70 358 L 70 378" />
                    <path id="prior-out2" d="M 255 332 L 255 358 L 200 358 L 200 378" />
                    <path id="prior-out3" d="M 275 332 L 275 358 L 330 358 L 330 378" />
                    <path id="prior-out4" d="M 300 332 L 300 358 L 460 358 L 460 378" />
                </defs>

                <!-- Connector lines -->
                <path d="M 70 62 L 70 108 L 265 108 L 265 128" class="path-line" />
                <path d="M 200 62 L 200 108 L 265 108 L 265 128" class="path-line" />
                <path d="M 330 62 L 330 108 L 265 108 L 265 128" class="path-line" />
                <path d="M 460 62 L 460 108 L 265 108 L 265 128" class="path-line" />
                <path d="M 240 172 L 200 172 L 200 198" class="path-line" />
                <path d="M 290 172 L 330 172 L 330 198" class="path-line" />
                <path d="M 200 242 L 200 268 L 265 268 L 265 288" class="path-line" />
                <path d="M 330 242 L 330 268 L 265 268 L 265 288" class="path-line" />
                <path d="M 230 332 L 230 358 L 70 358 L 70 378" class="path-line" />
                <path d="M 255 332 L 255 358 L 200 358 L 200 378" class="path-line" />
                <path d="M 275 332 L 275 358 L 330 358 L 330 378" class="path-line" />
                <path d="M 300 332 L 300 358 L 460 358 L 460 378" class="path-line" />

                <!-- Level 1: four inputs -->
                <g transform="translate(15, 20)">
                    <rect width="110" height="42" class="node-rect stroke-input" />
                    <text x="55" y="14" text-anchor="middle" class="text-title">Model</text>
                    <text x="55" y="27" text-anchor="middle" class="text-title">config</text>
                    <text x="55" y="38" text-anchor="middle" class="text-sub">JSON</text>
                </g>
                <g transform="translate(145, 20)">
                    <rect width="110" height="42" class="node-rect stroke-input" />
                    <text x="55" y="14" text-anchor="middle" class="text-title">Context</text>
                    <text x="55" y="27" text-anchor="middle" class="text-title">config</text>
                    <text x="55" y="38" text-anchor="middle" class="text-sub">JSON</text>
                </g>
                <g transform="translate(275, 20)">
                    <rect width="110" height="42" class="node-rect stroke-input" />
                    <text x="55" y="14" text-anchor="middle" class="text-title">MOJO</text>
                    <text x="55" y="27" text-anchor="middle" class="text-title">models</text>
                    <text x="55" y="38" text-anchor="middle" class="text-sub">.zip</text>
                </g>
                <g transform="translate(405, 20)">
                    <rect width="110" height="42" class="node-rect stroke-input" />
                    <text x="55" y="14" text-anchor="middle" class="text-title">Case</text>
                    <text x="55" y="27" text-anchor="middle" class="text-title">dataset</text>
                    <text x="55" y="38" text-anchor="middle" class="text-sub">CSV</text>
                </g>

                <!-- Level 2: processing -->
                <g transform="translate(195, 128)">
                    <rect width="140" height="44" class="node-rect stroke-process" />
                    <text x="70" y="16" text-anchor="middle" class="text-title">Analysis</text>
                    <text x="70" y="28" text-anchor="middle" class="text-title">pipeline</text>
                    <text x="70" y="39" text-anchor="middle" class="text-sub">ML inference</text>
                </g>

                <!-- Level 3: Ri & Ci -->
                <g transform="translate(140, 198)">
                    <rect width="120" height="44" class="node-rect stroke-ml" />
                    <text x="60" y="16" text-anchor="middle" class="text-title">Global ML</text>
                    <text x="60" y="28" text-anchor="middle" class="text-title">Risk</text>
                    <text x="60" y="39" text-anchor="middle" class="text-sub">Ri</text>
                </g>
                <g transform="translate(270, 198)">
                    <rect width="120" height="44" class="node-rect stroke-fuzzy" />
                    <text x="60" y="16" text-anchor="middle" class="text-title">Context</text>
                    <text x="60" y="28" text-anchor="middle" class="text-title">Align.</text>
                    <text x="60" y="39" text-anchor="middle" class="text-sub">Ci</text>
                </g>

                <!-- Level 4: prioritization -->
                <g transform="translate(185, 288)">
                    <rect width="160" height="44" class="node-rect stroke-hybrid" />
                    <text x="80" y="16" text-anchor="middle" class="text-title">Prioritization</text>
                    <text x="80" y="28" text-anchor="middle" class="text-title">Score</text>
                    <text x="80" y="39" text-anchor="middle" class="text-sub">λ·Ri + (1−λ)·Ci</text>
                </g>

                <!-- Level 5: analysis views -->
                <g transform="translate(15, 378)">
                    <rect width="110" height="42" class="node-rect stroke-output" />
                    <text x="55" y="14" text-anchor="middle" class="text-title">Overview</text>
                    <text x="55" y="26" text-anchor="middle" class="text-sub">prioritized</text>
                    <text x="55" y="36" text-anchor="middle" class="text-sub">cases</text>
                </g>
                <g transform="translate(145, 378)">
                    <rect width="110" height="42" class="node-rect stroke-output" />
                    <text x="55" y="14" text-anchor="middle" class="text-title">Models</text>
                    <text x="55" y="26" text-anchor="middle" class="text-sub">weights &amp;</text>
                    <text x="55" y="36" text-anchor="middle" class="text-sub">risks</text>
                </g>
                <g transform="translate(275, 378)">
                    <rect width="110" height="42" class="node-rect stroke-output" />
                    <text x="55" y="14" text-anchor="middle" class="text-title">Context</text>
                    <text x="55" y="26" text-anchor="middle" class="text-sub">rules &amp;</text>
                    <text x="55" y="36" text-anchor="middle" class="text-sub">membership</text>
                </g>
                <g transform="translate(405, 378)">
                    <rect width="110" height="42" class="node-rect stroke-output" />
                    <text x="55" y="14" text-anchor="middle" class="text-title">Robustness</text>
                    <text x="55" y="26" text-anchor="middle" class="text-sub">λ</text>
                    <text x="55" y="36" text-anchor="middle" class="text-sub">sensitivity</text>
                </g>

                <!-- Animated particles: inputs → processing -->
                <circle r="4" class="dot dot-input">
                    <animateMotion dur="8s" repeatCount="indefinite" calcMode="linear"
                                   keyTimes="0;0.25;1" keyPoints="0;1;1">
                        <mpath href="#in1"/>
                    </animateMotion>
                    <animate attributeName="opacity" values="1;1;1;0;0" dur="8s" repeatCount="indefinite"
                             keyTimes="0;0.02;0.23;0.25;1" />
                </circle>
                <circle r="4" class="dot dot-input">
                    <animateMotion dur="8s" repeatCount="indefinite" calcMode="linear"
                                   keyTimes="0;0.25;1" keyPoints="0;1;1">
                        <mpath href="#in2"/>
                    </animateMotion>
                    <animate attributeName="opacity" values="1;1;1;0;0" dur="8s" repeatCount="indefinite"
                             keyTimes="0;0.02;0.23;0.25;1" />
                </circle>
                <circle r="4" class="dot dot-input">
                    <animateMotion dur="8s" repeatCount="indefinite" calcMode="linear"
                                   keyTimes="0;0.25;1" keyPoints="0;1;1">
                        <mpath href="#in3"/>
                    </animateMotion>
                    <animate attributeName="opacity" values="1;1;1;0;0" dur="8s" repeatCount="indefinite"
                             keyTimes="0;0.02;0.23;0.25;1" />
                </circle>
                <circle r="4" class="dot dot-input">
                    <animateMotion dur="8s" repeatCount="indefinite" calcMode="linear"
                                   keyTimes="0;0.25;1" keyPoints="0;1;1">
                        <mpath href="#in4"/>
                    </animateMotion>
                    <animate attributeName="opacity" values="1;1;1;0;0" dur="8s" repeatCount="indefinite"
                             keyTimes="0;0.02;0.23;0.25;1" />
                </circle>

                <!-- Processing → Ri / Ci -->
                <circle r="4" class="dot dot-ml">
                    <animateMotion dur="8s" repeatCount="indefinite" calcMode="linear"
                                   keyTimes="0;0.27;0.40;1" keyPoints="0;0;1;1">
                        <mpath href="#proc-ri"/>
                    </animateMotion>
                    <animate attributeName="opacity" values="0;0;1;1;0;0" dur="8s" repeatCount="indefinite"
                             keyTimes="0;0.27;0.29;0.38;0.40;1" />
                </circle>
                <circle r="4" class="dot dot-fuzzy">
                    <animateMotion dur="8s" repeatCount="indefinite" calcMode="linear"
                                   keyTimes="0;0.29;0.42;1" keyPoints="0;0;1;1">
                        <mpath href="#proc-ci"/>
                    </animateMotion>
                    <animate attributeName="opacity" values="0;0;1;1;0;0" dur="8s" repeatCount="indefinite"
                             keyTimes="0;0.29;0.31;0.40;0.42;1" />
                </circle>

                <!-- Ri / Ci → Prioritization -->
                <circle r="5" class="dot dot-ml">
                    <animateMotion dur="8s" repeatCount="indefinite" calcMode="linear"
                                   keyTimes="0;0.52;0.70;1" keyPoints="0;0;1;1">
                        <mpath href="#ri-prior"/>
                    </animateMotion>
                    <animate attributeName="opacity" values="0;0;1;1;0;0" dur="8s" repeatCount="indefinite"
                             keyTimes="0;0.52;0.54;0.68;0.70;1" />
                </circle>
                <circle r="5" class="dot dot-fuzzy">
                    <animateMotion dur="8s" repeatCount="indefinite" calcMode="linear"
                                   keyTimes="0;0.54;0.72;1" keyPoints="0;0;1;1">
                        <mpath href="#ci-prior"/>
                    </animateMotion>
                    <animate attributeName="opacity" values="0;0;1;1;0;0" dur="8s" repeatCount="indefinite"
                             keyTimes="0;0.54;0.56;0.70;0.72;1" />
                </circle>

                <!-- Prioritization → outputs -->
                <circle r="4" class="dot dot-hybrid">
                    <animateMotion dur="8s" repeatCount="indefinite" calcMode="linear"
                                   keyTimes="0;0.74;0.96;1" keyPoints="0;0;1;1">
                        <mpath href="#prior-out1"/>
                    </animateMotion>
                    <animate attributeName="opacity" values="0;0;1;1;0;0" dur="8s" repeatCount="indefinite"
                             keyTimes="0;0.74;0.76;0.94;0.96;1" />
                </circle>
                <circle r="4" class="dot dot-output">
                    <animateMotion dur="8s" repeatCount="indefinite" calcMode="linear"
                                   keyTimes="0;0.75;0.97;1" keyPoints="0;0;1;1">
                        <mpath href="#prior-out2"/>
                    </animateMotion>
                    <animate attributeName="opacity" values="0;0;1;1;0;0" dur="8s" repeatCount="indefinite"
                             keyTimes="0;0.75;0.77;0.95;0.97;1" />
                </circle>
                <circle r="4" class="dot dot-output">
                    <animateMotion dur="8s" repeatCount="indefinite" calcMode="linear"
                                   keyTimes="0;0.76;0.98;1" keyPoints="0;0;1;1">
                        <mpath href="#prior-out3"/>
                    </animateMotion>
                    <animate attributeName="opacity" values="0;0;1;1;0;0" dur="8s" repeatCount="indefinite"
                             keyTimes="0;0.76;0.78;0.96;0.98;1" />
                </circle>
                <circle r="4" class="dot dot-output">
                    <animateMotion dur="8s" repeatCount="indefinite" calcMode="linear"
                                   keyTimes="0;0.77;0.99;1" keyPoints="0;0;1;1">
                        <mpath href="#prior-out4"/>
                    </animateMotion>
                    <animate attributeName="opacity" values="0;0;1;1;0;0" dur="8s" repeatCount="indefinite"
                             keyTimes="0;0.77;0.79;0.97;0.99;1" />
                </circle>
            </svg>
        </div>
    </body>
    </html>
    """
    components.html(html_code, height=465, scrolling=False)
