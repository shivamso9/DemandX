mermaid.initialize({ startOnLoad: false });

const getEl = (id) => document.getElementById(id);
const stages = { 1: getEl('stage-1'), '1a': getEl('stage-1a'), 2: getEl('stage-2'), 3: getEl('stage-3') };
const statuses = { 1: getEl('status-1'), '1a': getEl('status-1a'), 2: getEl('status-2'), 3: getEl('status-3') };
const contents = { 1: getEl('content-1'), '1a': getEl('content-1a'), 2: getEl('content-2'), 3: getEl('content-3') };
const fileUpload = getEl('file-upload'), fileNameDisplay = getEl('file-name'), validateBtn = getEl('validate-btn'), feedback1 = getEl('feedback-1'), resetBtn = getEl('reset-btn');

const API_BASE_URL = 'http://127.0.0.1:8080/api';
let isAutoMode = false;

let conversationHistory = [], generatedCode = { test: null, plugin_repo: null, plugin_tenant: null };
let currentLogicPlan = "";
let selectedReferencePlugin = null;
let proposedTestCases = [];
let parsedInputSchemas = null;
let extractedPluginName = null;
let customTestCode = null; // <-- ADDED: To store custom test file content
let testsToExclude = new Set();

const autoModeToggle = getEl('automode-toggle');
autoModeToggle.addEventListener('change', () => {
    isAutoMode = autoModeToggle.checked;
    console.log(`AutoMode is now ${isAutoMode ? 'ON' : 'OFF'}`);
});

function wordWrap(text, maxLength = 35) {
    if (!text || text.length <= maxLength) return text;
    const words = text.split(' ');
    let lines = [];
    let currentLine = '';
    for (const word of words) {
        if ((currentLine + ' ' + word).length > maxLength && currentLine.length > 0) {
            lines.push(currentLine.trim());
            currentLine = word;
        } else {
            currentLine += (currentLine ? ' ' : '') + word;
        }
    }
    if (currentLine) lines.push(currentLine.trim());
    return lines.join('<br>');
}

function addChatPanel(stage) {
    const stageContent = contents[stage];
    const existingPanel = getEl(`chat-panel-${stage}`);
    if (existingPanel) return;

    const chatPanel = document.createElement('div');
    chatPanel.id = `chat-panel-${stage}`;
    chatPanel.className = 'mt-4 border-t pt-4';
    chatPanel.innerHTML = `
        <button id="toggle-chat-btn-${stage}" class="text-sm font-semibold text-indigo-600 hover:text-indigo-800"><i class="fas fa-comments mr-2"></i>Refine With Agent</button>
        <div id="chat-box-${stage}" class="chat-container hidden mt-2">
            <div id="chat-history-${stage}" class="chat-history"></div>
            <div class="chat-input">
                <input type="text" id="user-chat-input-${stage}" placeholder="e.g., 'Change step 3 to group by region'...">
                <button id="send-chat-btn-${stage}"><i class="fas fa-paper-plane"></i></button>
            </div>
        </div>
    `;
    stageContent.appendChild(chatPanel);

    getEl(`toggle-chat-btn-${stage}`).addEventListener('click', () => {
        getEl(`chat-box-${stage}`).classList.toggle('hidden');
    });

    const chatInput = getEl(`user-chat-input-${stage}`);
    getEl(`send-chat-btn-${stage}`).addEventListener('click', () => handleChat(stage, chatInput));
    chatInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') handleChat(stage, chatInput);
    });
}

async function handleChat(stage, inputElement) {
    const chatHistory = getEl(`chat-history-${stage}`);
    const userMessage = inputElement.value.trim();
    if (!userMessage) return;

    chatHistory.innerHTML += `<div class="chat-message user">${userMessage}</div>`;
    inputElement.value = '';
    chatHistory.scrollTop = chatHistory.scrollHeight;

    const response = await callBackend(`${API_BASE_URL}/chat`, {
        stage: stage,
        message: userMessage,
        context: {
            logic_summary: currentLogicPlan,
            proposed_tests: proposedTestCases
        }
    });

    const agentMessage = response.text || "Sorry, I couldn't process that request.";

    if (stage === 1 && (agentMessage.includes("### PSEUDOCODE ###") || agentMessage.includes("### FLOWCHART ###"))) {
        currentLogicPlan = agentMessage;
        renderApprovalUI(currentLogicPlan);
    } else {
        chatHistory.innerHTML += `<div class="chat-message agent">${agentMessage}</div>`;
    }

    chatHistory.scrollTop = chatHistory.scrollHeight;
}

function extractCodeParts(responseText) {
    const cleanCode = (text) => text ? text.trim().replace(/^\s*```(python|py)?\s*|\s*```\s*$/g, '') : '';
    const tenantMatch = responseText.match(/### TENANT FILE ###\s*([\s\S]*?)### REPO FILE ###/);
    const repoMatch = responseText.match(/### REPO FILE ###\s*([\s\S]*)/);

    const plugin_tenant = tenantMatch ? cleanCode(tenantMatch[1]) : "### ERROR: TENANT FILE marker not found. ###";
    const plugin_repo = repoMatch ? cleanCode(repoMatch[1]) : "### ERROR: REPO FILE marker not found. ###";

    return { plugin_tenant, plugin_repo };
}


async function callBackend(url, payload = {}, method = 'POST', isFileUpload = false) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 1800000); 

    try {
        const options = { 
            method,
            signal: controller.signal 
        };
        if (method !== 'GET') {
            if (isFileUpload) {
                options.body = payload;
            } else {
                options.headers = { 'Content-Type': 'application/json' };
                options.body = JSON.stringify(payload);
            }
        }
        const response = await fetch(url, options);
        clearTimeout(timeoutId);
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({ error: `Server returned ${response.status} ${response.statusText}` }));
            const msg = errorData.error || `API Error (${response.status} ${response.statusText}).`;
            if (response.status === 500) return { error: `Backend Error (500): The server encountered a critical error. Check the terminal where \`backend.py\` is running for the full traceback.` };
            throw new Error(msg);
        }
        return await response.json();
    } catch (error) {
        clearTimeout(timeoutId);
        console.error("Backend call failed:", error);
        if (error.name === 'AbortError') {
            return { error: `Error: **Request timed out.** The AI model took too long to respond (more than 30 minutes). Please try again.` };
        }
        if (error instanceof TypeError) return { error: `Error: **Connection to backend failed.** Please ensure the server is running and check the terminal for errors.` };
        return { error: `Error: ${error.message}` };
    }
}

async function handleValidation() {
    setStatus(1, 'running');
    const file = fileUpload.files[0];
    if (!file) {
        setStatus(1, 'failed', 'Please select a file first.');
        return;
    }
    const formData = new FormData();
    formData.append('file', file);
    const response = await callBackend(`${API_BASE_URL}/validate`, formData, 'POST', true);

    if (response.error) {
         setStatus(1, 'failed', response.error);
         return;
    }

    parsedInputSchemas = response.parsed_schemas;
    extractedPluginName = response.plugin_name; 
    console.log(`Plugin name from Excel: ${extractedPluginName}`);
    const responseText = response.text || "";

    if (responseText.startsWith('VALIDATION_FAIL:')) {
        setStatus(1, 'failed', responseText.replace('VALIDATION_FAIL:', '').trim());
    } else if (responseText.startsWith('VALIDATION_SUCCESS:')) {
        currentLogicPlan = responseText.replace('VALIDATION_SUCCESS:', '').trim();
        renderApprovalUI(currentLogicPlan);
         if (isAutoMode) {
            console.log("AutoMode: Auto-approving Stage 1.");
            setTimeout(handleApproval, 100); 
        }
    } else {
        setStatus(1, 'failed', 'Received an unexpected response from the validation agent. Check the backend for details.');
    }
}

function setStatus(stage, type, message = '') {
    const statusEl = statuses[stage], stageEl = stages[stage];
    stageEl.classList.remove('active', 'completed', 'failed');
    const icons = { pending: '<i class="fas fa-clock mr-2"></i>Pending', running: '<div class="thinking-dots"><span>.</span><span>.</span><span>.</span></div><span class="ml-2">Thinking...</span>', success: '<i class="fas fa-check-circle mr-2"></i>Success', failed: '<i class="fas fa-times-circle mr-2"></i>Action Required' };
    const colors = { pending: 'text-gray-500', running: 'text-blue-500', success: 'text-green-500', failed: 'text-red-500' };

    if (type === 'active' || type === 'running') stageEl.classList.add('active');
    if (type === 'success') stageEl.classList.add('completed');
    if (type === 'failed') stageEl.classList.add('failed');

    statusEl.innerHTML = `<div class="flex items-center font-medium ${colors[type]}">${icons[type]}</div>`;

    if (message) {
        const feedbackEl = contents[stage];
        feedbackEl.innerHTML = `<div class="text-red-700 p-4 bg-red-100 rounded-lg font-semibold">${message.replace(/\*\*/g, '<strong>')}</div>`;
    }
}


function activateStage(stage) {
    if(stages[stage]) {
        stages[stage].open = true;
    }
    stages[stage].classList.remove('opacity-50');
    const icon = stages[stage].querySelector('h2 > i.fas');
    if(icon) {
        icon.classList.remove('text-gray-400');
        icon.classList.add('text-indigo-500');
    }
    setStatus(stage, 'pending');
    contents[stage].classList.remove('hidden');
}

function renderApprovalUI(planText) {
    setStatus(1, 'failed');
    feedback1.classList.remove('hidden');

    const pseudoCodeMatch = planText.match(/### PSEUDOCODE ###\s*([\s\S]*?)### FLOWCHART ###/);
    const flowchartMatch = planText.match(/### FLOWCHART ###\s*([\s\S]*?)### OUTPUT SCHEMA ###/);
    const outputSchemaMatch = planText.match(/### OUTPUT SCHEMA ###\s*([\s\S]*)/);

    const pseudoCode = pseudoCodeMatch ? pseudoCodeMatch[1].trim() : "Could not generate pseudocode.";

    let flowchartHTML = '<div class="text-gray-500">Could not generate flowchart.</div>';
    if (flowchartMatch) {
        const flowchartSteps = flowchartMatch[1].trim().split('\n').filter(s => s.trim().match(/^\d+\./));
        if (flowchartSteps.length > 0) {
            let mermaidCode = 'graph TD;';
            mermaidCode += '\n  classDef default fill:#f0f6ff,stroke:#a0c8ff,stroke-width:1px,color:#333;';
            const nodeIds = flowchartSteps.map((_, index) => 'N' + index);
            flowchartSteps.forEach((step, index) => {
                const cleanText = step.replace(/^\d+\.\s*/, '').trim().replace(/"/g, '#quot;');
                mermaidCode += `\n  ${nodeIds[index]}("${wordWrap(cleanText)}");`;
            });
            for (let i = 1; i < nodeIds.length; i++) {
                mermaidCode += `\n  ${nodeIds[i-1]} --> ${nodeIds[i]};`;
            }
            flowchartHTML = `<pre class="mermaid">${mermaidCode}</pre>`;
        }
    }

    feedback1.innerHTML = `
        <div class="p-4 bg-indigo-50 rounded-lg border border-indigo-200" id="plan-container-stage1">
            <h3 class="font-semibold text-lg text-indigo-800 mb-4"><i class="fas fa-lightbulb mr-2"></i>Agent's Understanding & Plan</h3>
            <div class="grid md:grid-cols-2 gap-6">
                <div>
                    <h4 class="font-semibold text-gray-700 mb-3 border-b pb-2">Pseudocode</h4>
                    <pre class="text-sm whitespace-pre-wrap text-gray-600 leading-relaxed">${pseudoCode}</pre>
                </div>
                <div>
                    <h4 class="font-semibold text-gray-700 mb-3 border-b pb-2">Flowchart</h4>
                    ${flowchartHTML}
                </div>
            </div>
        </div>
        <div id="approval-actions" class="flex items-center justify-end mt-4 space-x-3">
            <button id="reject-btn" class="bg-red-500 hover:bg-red-600 text-white font-bold py-2 px-4 rounded-lg transition"><i class="fas fa-times mr-2"></i>Reject</button>
            <button id="approve-btn" class="bg-green-500 hover:bg-green-600 text-white font-bold py-2 px-4 rounded-lg transition"><i class="fas fa-check mr-2"></i>Approve & Proceed</button>
        </div>
    `;

    setTimeout(() => mermaid.run(), 0);

    addChatPanel(1);
    getEl('approve-btn').addEventListener('click', handleApproval);
    getEl('reject-btn').addEventListener('click', () => setStatus(1, 'failed', 'Plan rejected. Please refine your request or upload a new file.'));
}

async function handleApproval() {
    setStatus(1, 'running');
    const planContainerHTML = getEl('plan-container-stage1')?.outerHTML || '<p>Plan content was not found.</p>';
    getEl('approval-actions').innerHTML = `<p class="text-gray-600">Checking plan...</p>`;

    const planResponse = await callBackend(`${API_BASE_URL}/plan`, { parsed_schemas: parsedInputSchemas });

    if (planResponse.status === "PLAN_FAIL") {
        setStatus(1, 'failed', planResponse.reason);
        renderApprovalUI(currentLogicPlan);
        return;
    }

    setStatus(1, 'success');
    stages[1].open = false; 
    feedback1.innerHTML = `
        <details class="bg-white rounded-lg shadow">
            <summary class="cursor-pointer p-4 font-semibold text-gray-700 hover:bg-gray-50 flex justify-between items-center">
                <span><i class="fas fa-check-circle mr-2 text-green-500"></i>Approved Plan (Click to toggle)</span>
                <i class="fas fa-chevron-down transition-transform"></i>
            </summary>
            <div class="p-4 border-t">${planContainerHTML}</div>
        </details>`;
    runStage1a_SelectReference();
}

async function runStage1a_SelectReference() {
    activateStage('1a');
    setStatus('1a', 'running');

    const response = await callBackend(`${API_BASE_URL}/list_plugins`, {}, 'GET');

    if (response.error) {
        setStatus('1a', 'failed', `Could not load reference plugins: ${response.error}`);
        return;
    }
    
    const plugins = response.plugins || [];
    if (isAutoMode && plugins.length > 0) {
        const DEFAULT_REFERENCE_PLUGIN = 'DP009OutlierCleansing'
        const pluginToSelect = plugins.includes(DEFAULT_REFERENCE_PLUGIN) 
        ? DEFAULT_REFERENCE_PLUGIN 
        : plugins[0];
        console.log(`AutoMode: Auto-selecting reference plugin: ${pluginToSelect}`);
        selectedReferencePlugin = pluginToSelect;
        contents['1a'].innerHTML = `<p class="text-indigo-600"><i class="fas fa-robot mr-2"></i>AutoMode selected reference: <strong>${selectedReferencePlugin}</strong>. Proceeding...</p>`;
        setStatus('1a', 'success');
        stages['1a'].open = false;
        checkAndLoadExistingTestPlan();
        return; 
    }

    setStatus('1a', 'failed'); // Action Required
    contents['1a'].innerHTML = `
        <p class="text-gray-600 mb-4">Select a reference plugin to guide the Coder Agent.</p>
        <div class="relative">
            <input type="text" id="plugin-search-input" class="w-full border-gray-300 rounded-lg shadow-sm p-2" placeholder="Search for a plugin...">
            <div id="plugin-search-list" class="absolute z-10 w-full bg-white border rounded-lg mt-1 shadow-lg hidden">
                ${plugins.map(p => `<div class="p-2 cursor-pointer hover:bg-indigo-100" data-plugin-name="${p}">${p}</div>`).join('')}
            </div>
        </div>
        <p id="selected-plugin-display" class="mt-4 font-semibold text-indigo-600"></p>
        <button id="confirm-plugin-btn" class="mt-4 bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg transition hidden"><i class="fas fa-arrow-right mr-2"></i>Confirm & Continue</button>
    `;
    addChatPanel('1a');

    const searchInput = getEl('plugin-search-input'), searchList = getEl('plugin-search-list');
    const display = getEl('selected-plugin-display'), confirmBtn = getEl('confirm-plugin-btn');

    searchInput.addEventListener('focus', () => searchList.classList.remove('hidden'));
    document.addEventListener('click', (e) => {
        if (!getEl('content-1a').contains(e.target)) searchList.classList.add('hidden');
    });

    searchInput.addEventListener('keyup', () => {
        const filter = searchInput.value.toLowerCase();
        searchList.querySelectorAll('div').forEach(item => {
            item.style.display = item.textContent.toLowerCase().includes(filter) ? '' : 'none';
        });
    });

    searchList.querySelectorAll('div').forEach(item => {
        item.addEventListener('click', () => {
            selectedReferencePlugin = item.dataset.pluginName;
            display.innerHTML = `<i class="fas fa-check-circle text-green-500 mr-2"></i>Selected: <strong>${selectedReferencePlugin}</strong>`;
            confirmBtn.classList.remove('hidden');
            searchInput.value = selectedReferencePlugin;
            searchList.classList.add('hidden');
        });
    });

    confirmBtn.addEventListener('click', () => {
        setStatus('1a', 'success');
        stages['1a'].open = false;
        checkAndLoadExistingTestPlan();
    });
}

async function checkAndLoadExistingTestPlan() {
    activateStage(2);
    setStatus(2, 'running');
    
    // Use GET method for retrieving data
    const response = await callBackend(`${API_BASE_URL}/get_test_plan/${extractedPluginName}`, {}, 'GET');

    if (response.status === 'found') {
        setStatus(2, 'failed'); // Action Required
        contents[2].innerHTML = `
            <div class="p-4 bg-indigo-50 rounded-lg border border-indigo-200">
                <h3 class="font-semibold text-lg text-indigo-800 mb-2"><i class="fas fa-history mr-2"></i>Existing Test Plan Found</h3>
                <p class="text-gray-600 mb-4">We found a previously saved test plan for <strong>${extractedPluginName}</strong>. What would you like to do?</p>
                <div class="flex items-center space-x-4">
                    <button id="reuse-plan-btn" class="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg transition"><i class="fas fa-recycle mr-2"></i>Reuse Existing Plan</button>
                    <button id="generate-new-plan-btn" class="bg-gray-600 hover:bg-gray-700 text-white font-bold py-2 px-4 rounded-lg transition"><i class="fas fa-magic mr-2"></i>Generate New Plan</button>
                </div>
            </div>`;

        getEl('reuse-plan-btn').addEventListener('click', () => {
            proposedTestCases = response.plan;
            proposedTestCases.forEach(tc => tc.checked = true); // Ensure all are checked by default
            renderTestSelectionUI();
        });
        getEl('generate-new-plan-btn').addEventListener('click', runStage2_TestArchitect);
    } else {
        // No plan found, proceed to generate a new one
        runStage2_TestArchitect();
    }
}

async function runStage2_TestArchitect() {
    activateStage(2);
    setStatus(2, 'running');
    contents[2].innerHTML = `<p>Test Architect Agent is designing a comprehensive test suite...</p>`;

    const prompt = `You are a Test Architect Agent. Based on the business logic, propose a list of pytest test cases.
    Business Logic: "${currentLogicPlan}"
    **CRITICAL INSTRUCTIONS:**
    1.  Propose a comprehensive suite of tests covering "Golden Record Validation", "Edge Cases", and "Extrapolation/New Scenarios".
    2.  For each test case, provide a concise 'name' (e.g., "test_golden_record"), a 'description', and the 'category'.
    3.  Your entire output **MUST** be only the raw JSON array of objects. Do not include any other text, explanations, or markdown code blocks.
    4.  **STRICT JSON FORMAT:** All property names and string values MUST be enclosed in double quotes (").
    **EXAMPLE JSON OUTPUT:**
    [
        { "category": "Golden Record Validation", "name": "test_golden_record_validation", "description": "Validates the main success path with standard inputs." },
        { "category": "Edge Cases", "name": "test_null_or_empty_input", "description": "Tests graceful exit when a primary input is null or empty." }
    ]`;

    const response = await callBackend(`${API_BASE_URL}/generate`, { prompt });

    if(response.error) {
        setStatus(2, 'failed', response.error);
        return;
    }
    
    const parseAndProceed = (jsonString) => {
        proposedTestCases = JSON.parse(jsonString);
        proposedTestCases.forEach(tc => tc.checked = true);
        
        renderTestSelectionUI(isAutoMode);

        if (isAutoMode) {
            console.log("AutoMode: UI rendered, now auto-generating all proposed tests.");
            setTimeout(handleGenerateSelectedTests, 100);
        }
    };

    try {
        parseAndProceed(response.text);
    } catch (e) {
        console.warn("Direct JSON parsing failed, attempting to find JSON block.", e);
        const jsonMatch = response.text.match(/\[\s*\{[\s\S]*\}\s*\]/);
        try {
            if (!jsonMatch) { throw new Error("Could not find a valid JSON array in the AI's response."); }
            parseAndProceed(jsonMatch[0]);
        } catch (e2) {
            console.error("Could not parse extracted JSON, calling fixer agent.", e2);
            contents[2].innerHTML += `<p class="text-orange-500 mt-2">Initial parsing failed. Attempting to self-correct...</p>`;
            const fixResponse = await callBackend(`${API_BASE_URL}/fix_json`, { bad_json: response.text });
            if (fixResponse.error) {
                setStatus(2, 'failed', `Failed to parse test plan, and the fixer agent also failed. Error: ${fixResponse.error}`);
                contents[2].innerHTML += `<h4 class="font-semibold text-gray-700 mt-4">Original AI Response:</h4><pre class="code-block">${escapeHtml(response.text)}</pre>`;
            } else {
                try {
                    parseAndProceed(fixResponse.fixed_json);
                } catch (e3) {
                     setStatus(2, 'failed', `Failed to parse test plan even after self-correction. Error: ${e3}`);
                     contents[2].innerHTML += `<h4 class="font-semibold text-gray-700 mt-4">Original AI Response:</h4><pre class="code-block">${escapeHtml(response.text)}</pre><h4 class="font-semibold text-gray-700 mt-4">Corrected AI Response:</h4><pre class="code-block">${escapeHtml(fixResponse.fixed_json)}</pre>`;
                }
            }
        }
    }
}

function updateTestSelectionState(index, isChecked) {
    if (proposedTestCases[index]) {
        proposedTestCases[index].checked = isChecked;
    }
}

function renderTestSelectionUI(isAuto = false) {
    if (!isAuto) {
        setStatus(2, 'failed'); 
    }
    const categories = [...new Set(proposedTestCases.map(tc => tc.category || "General"))];
    let testCasesHtml = categories.map(category => `
        <details class="bg-white rounded-lg shadow mb-4" open>
            <summary class="cursor-pointer p-4 font-semibold text-gray-700 hover:bg-gray-50 flex justify-between items-center">
                <span>${escapeHtml(category)}</span>
                <i class="fas fa-chevron-down transition-transform"></i>
            </summary>
            <div id="test-category-${category.replace(/[^a-zA-Z0-9]/g, '-')}" class="p-4 border-t space-y-2">
            ${proposedTestCases
                .map((tc, index) => ({...tc, originalIndex: index}))
                .filter(tc => (tc.category || "General") === category)
                .map(tc => `
                <div class="test-case-item border rounded-lg p-3 flex items-start space-x-3">
                    <input type="checkbox" onchange="updateTestSelectionState(${tc.originalIndex}, this.checked)" id="test_case_${tc.originalIndex}" name="test_case" value="${tc.originalIndex}" class="mt-1 h-4 w-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500" ${tc.checked ? 'checked' : ''}>
                    <div class="flex-grow">
                        <strong contenteditable="true" onblur="updateTestCase(this, ${tc.originalIndex}, 'name')" class="text-gray-800 focus:outline-blue-500 focus:bg-white rounded px-1">${tc.name}</strong><br>
                        <p contenteditable="true" onblur="updateTestCase(this, ${tc.originalIndex}, 'description')" class="text-sm text-gray-600 mt-1 focus:outline-blue-500 focus:bg-white rounded px-1">${tc.description}</p>
                    </div>
                    <button onclick="deleteTestCase(${tc.originalIndex})" class="text-gray-400 hover:text-red-500 transition-colors"><i class="fas fa-trash-alt"></i></button>
                </div>
            `).join('')}
            </div>
        </details>
    `).join('');

    const testPlanContainer = getEl('test-plan-container');
    if (testPlanContainer && getEl('test-case-list')) {
         getEl('test-case-list').innerHTML = testCasesHtml;
    } else {
        contents[2].innerHTML = `
            <div id="test-plan-container">
                <div class="p-4 bg-indigo-50 rounded-lg border border-indigo-200">
                    <h3 class="font-semibold text-xl text-indigo-800 mb-2"><i class="fas fa-clipboard-list mr-2"></i>Proposed Test Plan</h3>
                    <p class="text-gray-600 mb-4">Review, edit, add, or delete the test cases below.</p>
                    <div id="test-case-list">${testCasesHtml}</div>
                    <div class="flex items-center justify-between mt-6 border-t pt-4">
                        <div>
                            <button id="select-all-btn" class="text-sm font-medium text-indigo-600 hover:text-indigo-800">Select All</button> |
                            <button id="deselect-all-btn" class="text-sm font-medium text-indigo-600 hover:text-indigo-800">Deselect All</button> |
                            <button id="add-test-case-btn" class="text-sm font-medium text-indigo-600 hover:text-indigo-800 ml-2"><i class="fas fa-plus mr-1"></i> Add New Test Case</button>
                        </div>
                        <button id="generate-tests-btn" class="bg-green-500 hover:bg-green-600 text-white font-bold py-2 px-4 rounded-lg transition"><i class="fas fa-cogs mr-2"></i>Generate Selected Tests</button>
                    </div>
                </div>
            </div>
            <div id="generated-code-container" class="mt-4"></div>
        `;
        addChatPanel(2);

        getEl('generate-tests-btn').addEventListener('click', handleGenerateSelectedTests);
        getEl('select-all-btn').addEventListener('click', () => {
            proposedTestCases.forEach(tc => tc.checked = true);
            renderTestSelectionUI();
        });
        getEl('deselect-all-btn').addEventListener('click', () => {
            proposedTestCases.forEach(tc => tc.checked = false);
            renderTestSelectionUI();
        });
        getEl('add-test-case-btn').addEventListener('click', addNewTestCase);
    }
}

function updateTestCase(element, index, field) {
    if (proposedTestCases[index]) {
        proposedTestCases[index][field] = element.innerText;
    }
    console.log(`Updated test case ${index}:`, proposedTestCases[index]);
}

function deleteTestCase(index) {
    if (confirm('Are you sure you want to delete this test case?')) {
        proposedTestCases.splice(index, 1);
        renderTestSelectionUI();
    }
}

function addNewTestCase() {
    proposedTestCases.push({
        category: 'Custom',
        name: `new_test_case_${Date.now()}`,
        description: 'Click to edit description',
        checked: true
    });
    renderTestSelectionUI();
}

function exportTestResultsToCSV(attemptDetails, pluginName) {
    const lastAttempt = attemptDetails[attemptDetails.length - 1];
    if (!lastAttempt) return;

    // Helper to format data for CSV cells
    const formatCell = (data) => {
        if (!data) return '""';
        // Escape double quotes by doubling them and wrap everything in double quotes
        const str = String(data).replace(/"/g, '""');
        return `"${str}"`;
    };

    let csvContent = "";
    // --- MODIFIED: New headers ---
    const headers = [
        "Test Name", "Category", "Description (The 'Why')", "Status", 
        "Input Data", "Expected Output", "Result / Failure Details"
    ];
    csvContent += headers.join(",") + "\r\n";

    const rows = [];
    lastAttempt.passed.forEach(test => {
        const originalTest = proposedTestCases.find(tc => tc.name === test.name.split('[')[0]);
        const category = originalTest ? originalTest.category : 'N/A';
        rows.push([
            formatCell(test.name),
            formatCell(category),
            formatCell(test.description),
            formatCell("Passed"),
            formatCell(test.input_data),
            formatCell(test.expected_data),
            formatCell("Actual output matched expected output.")
        ].join(","));
    });

    lastAttempt.failed.forEach(test => {
        const originalTest = proposedTestCases.find(tc => tc.name === test.name.split('[')[0]);
        const category = originalTest ? originalTest.category : 'N/A';
        rows.push([
            formatCell(test.name),
            formatCell(category),
            formatCell(test.description),
            formatCell("Failed"),
            formatCell(test.input_data),
            formatCell(test.expected_data),
            formatCell(test.reason) // The detailed failure reason
        ].join(","));
    });

    csvContent += rows.join("\r\n");

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement("a");
    const url = URL.createObjectURL(blob);
    link.setAttribute("href", url);
    link.setAttribute("download", `${pluginName}_Release_Report_${new Date().toISOString().slice(0,10)}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

async function handleGenerateSelectedTests() {
    const selectedTests = isAutoMode ? proposedTestCases : proposedTestCases.filter(tc => tc.checked);

    if (selectedTests.length === 0) {
        alert("Please select at least one test case to generate.");
        return;
    }

    // Save the plan as before
    if (extractedPluginName) {
        await callBackend(`${API_BASE_URL}/save_test_plan`, {
            plugin_name: extractedPluginName,
            test_plan: proposedTestCases
        });
        console.log(`Test plan for ${extractedPluginName} saved.`);
    }

    setStatus(2, 'running');
    
    // ... (UI loading message code remains the same) ...
    
    // MODIFICATION: Call the new chunked endpoint with a different payload
    const payload = {
        logic_summary: currentLogicPlan,
        test_cases: selectedTests
    };

    // Note the new URL: /api/generate_test_file
    const response = await callBackend(`${API_BASE_URL}/generate_test_file`, payload);

    // ... (rest of the function remains the same, handling the response) ...

    if (response.error) {
        setStatus(2, 'failed', response.error);
    } else {
        const code = response.text.replace(/```(python|py)?/g, '').replace(/```/g, '');
        generatedCode['test'] = code;
        
        if (isAutoMode) {
            console.log("AutoMode: Pytest code generated, proceeding to Stage 3.");
            displayGeneratedTestCode(code);
            setTimeout(() => getEl('approve-tests-btn')?.click(), 100);
        } else {
            displayGeneratedTestCode(code);
        }
    }
}

function displayGeneratedTestCode(code) {
    setStatus(2, 'success');
    
    const testPlanContainer = getEl('test-plan-container');
    if (testPlanContainer && testPlanContainer.tagName !== 'DETAILS') {
        const planContentHTML = testPlanContainer.innerHTML;
        testPlanContainer.outerHTML = `
            <details id="test-plan-container" class="mb-4" open>
                <summary class="cursor-pointer p-4 font-semibold text-gray-700 hover:bg-gray-50 flex justify-between items-center bg-white rounded-lg shadow">
                    <span><i class="fas fa-clipboard-list mr-2 text-indigo-500"></i>View/Edit Test Plan (Click to toggle)</span>
                    <i class="fas fa-chevron-down transition-transform"></i>
                </summary>
                <div class="mt-2">${planContentHTML}</div>
            </details>
        `;
        getEl('generate-tests-btn').addEventListener('click', handleGenerateSelectedTests);
        getEl('select-all-btn').addEventListener('click', () => {
            proposedTestCases.forEach(tc => tc.checked = true);
            renderTestSelectionUI();
        });
        getEl('deselect-all-btn').addEventListener('click', () => {
            proposedTestCases.forEach(tc => tc.checked = false);
            renderTestSelectionUI();
        });
        getEl('add-test-case-btn').addEventListener('click', addNewTestCase);
    }
    
    const codeContainer = getEl('generated-code-container');
    codeContainer.innerHTML = `
        <h3 class="font-semibold text-xl text-gray-800 mb-2">Generated Pytest File</h3>
        <p class="text-sm text-gray-600 mb-2">You can now edit the generated test code directly below.</p>
        <textarea id="generated-test-code-editor" class="w-full h-96 font-mono text-sm border rounded-lg p-2 bg-gray-900 text-gray-200">${escapeHtml(code)}</textarea>

        <div class="mt-6 pt-6 border-t">
             <h3 class="font-semibold text-xl text-gray-800 mb-2">Upload Additional Pytest File (Optional)</h3>
             <p class="text-sm text-gray-600 mb-2">You can upload an additional file with your own pytest test cases. It will be executed alongside the generated tests.</p>
             <div class="flex items-center space-x-4">
                 <label for="custom-test-upload" class="cursor-pointer inline-block bg-gray-500 hover:bg-gray-600 text-white font-bold py-2 px-4 rounded-lg transition"><i class="fas fa-upload mr-2"></i>Choose File</label>
                 <input id="custom-test-upload" type="file" class="hidden" accept=".py">
                 <p id="custom-test-filename" class="text-gray-600 font-semibold"></p>
            </div>
        </div>

        <div class="flex justify-end mt-6">
            <button id="approve-tests-btn" class="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg transition">
                Approve & Proceed to Coder Agent <i class="fas fa-arrow-right ml-2"></i>
            </button>
        </div>
    `;

    getEl('custom-test-upload').addEventListener('change', (event) => {
        const file = event.target.files[0];
        if (!file) {
            customTestCode = null;
            getEl('custom-test-filename').textContent = '';
            return;
        }
        getEl('custom-test-filename').textContent = `✅ ${file.name}`;
        const reader = new FileReader();
        reader.onload = (e) => {
            customTestCode = e.target.result;
            console.log("Custom test file loaded.");
        };
        reader.readAsText(file);
    });

    getEl('approve-tests-btn').addEventListener('click', () => {
        // Update the generated code with any edits before proceeding
        generatedCode.test = getEl('generated-test-code-editor').value;
        stages[2].open = false;
        runStage3_Coder();
    });
}

async function runStage3_Coder() {
    testsToExclude.clear();
    activateStage(3);
    const coderInitView = document.createElement('div');
    coderInitView.id = 'coder-init-view';
    contents[3].innerHTML = ''; 
    contents[3].appendChild(coderInitView);

    coderInitView.innerHTML = `
        <p class="mb-4">The Coder Agent will now generate the plugin files. The Debugger Agent will then automatically test and attempt to fix any issues.</p>
        <button id="generate-and-test-btn" class="bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg transition"><i class="fas fa-play-circle mr-2"></i>Generate & Test Plugin</button>
    `;
    addChatPanel(3);

    if (isAutoMode) {
         console.log("AutoMode: Auto-generating and testing plugin.");
         generateAndTestCode();
    } else {
        getEl('generate-and-test-btn').addEventListener('click', generateAndTestCode);
    }
}

async function generateAndTestCode() {
    setStatus(3, 'running');
    const coderInitView = getEl('coder-init-view');
    
    if (coderInitView) {
        coderInitView.innerHTML = `<div class="p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <p class="font-semibold text-blue-800 flex items-center">
                <i class="fas fa-robot fa-spin mr-3 text-xl"></i>
                <span><strong>Step 1/2: Coder Agent</strong> is generating the plugin files...</span>
            </p>
        </div>`;
    }

    const generatePrompt = `You are a Coder Agent. Your task is to generate two Python files (a Tenant file and a Repo file) based on the provided business logic and mimicking the provided reference files.
    **CRITICAL INSTRUCTIONS:**
    1.  **Function Signature:** The Repo file's main function signature MUST accept multiple pandas DataFrames as separate arguments, just like the reference files (e.g., \`def main(ItemMaster, AssortmentFinal, ...):\`). DO NOT use a single dictionary as input.
    2.  **Generate Two Files:** Your output must contain two distinct, clearly marked sections: \`### TENANT FILE ###\` and \`### REPO FILE ###\`.
    3.  **Tenant File:** This file should get each DataFrame from the data lake and pass them as separate arguments to the repo's main function.
    4.  **Repo File:** This file contains all the core business logic.
    ---
    **Business Logic to Implement:**
    ---
    ${currentLogicPlan}
    ---
    **STRICT OUTPUT FORMAT:** Your response must contain ONLY the two markers and their corresponding code blocks. No other text.`;

    const generateResponse = await callBackend(`${API_BASE_URL}/generate`, {
        prompt: generatePrompt,
        reference_plugin: selectedReferencePlugin
    });

    if (generateResponse.error) {
        setStatus(3, 'failed', generateResponse.error);
        if (coderInitView) coderInitView.innerHTML = '';
        return;
    }

    const { plugin_tenant, plugin_repo } = extractCodeParts(generateResponse.text);
    if (plugin_tenant.includes("### ERROR:") || plugin_repo.includes("### ERROR:")) {
        setStatus(3, 'failed', `Coder Agent output parsing failed. Check console for details.`);
        if (coderInitView) coderInitView.innerHTML = '';
        return;
    }
    
    generatedCode.plugin_repo = plugin_repo;
    generatedCode.plugin_tenant = plugin_tenant;

    if (coderInitView) {
        coderInitView.innerHTML = `<div class="p-4 bg-blue-50 border border-blue-200 rounded-lg">
            <p class="font-semibold text-blue-800 flex items-center">
                <i class="fas fa-robot fa-spin mr-3 text-xl"></i>
                <span><strong>Step 2/2: Debugger Agent</strong> is executing tests and attempting fixes...</span>
            </p>
        </div>`;
    }

    const selectedTests = proposedTestCases.filter(tc => tc.checked);
    const baseFileName = fileNameDisplay.textContent.replace(/\.xlsx?$/, '');
    const defaultPluginName = baseFileName ? `DP0xx${baseFileName}` : 'DP0xxNewPlugin';

    const executePayload = {
        test_code: generatedCode.test,
        custom_test_code: customTestCode, // <-- ADDED: Send custom test code
        plugin_repo_code: plugin_repo,
        plugin_tenant_code: plugin_tenant,
        logic_summary: currentLogicPlan,
        plugin_name: extractedPluginName || defaultPluginName,
        proposed_test_cases: selectedTests,
    };

    const executeResponse = await callBackend(`${API_BASE_URL}/execute`, executePayload);
    if (coderInitView) coderInitView.remove();

    if (executeResponse.error) {
        setStatus(3, 'failed', executeResponse.error);
        return;
    }
    
    generatedCode.plugin_repo = executeResponse.plugin_repo;
    generatedCode.plugin_tenant = executeResponse.plugin_tenant;
    generatedCode.test = executeResponse.test_code;

    displayFinalReport(executeResponse);
}

async function handleRerunDebugger() {
    setStatus(3, 'running');
    contents[3].innerHTML = `<div class="p-4 bg-blue-50 border border-blue-200 rounded-lg">
        <p class="font-semibold text-blue-800 flex items-center">
            <i class="fas fa-robot fa-spin mr-3 text-xl"></i>
            <span>Re-running the Debugger Agent with updated test configuration...</span>
        </p>
    </div>`;
    
    const testsToRemove = testsToExclude;
    console.log("Rerunning debugger, removing these tests:", Array.from(testsToRemove));

    const lines = generatedCode.test.split('\n');
    const newLines = [];
    let inFunctionToComment = false;

    for (const line of lines) {
        const isNewFunction = /^\s*def\s+/.test(line);

        if (isNewFunction) {
            const funcMatch = line.match(/^\s*def\s+([a-zA-Z0-9_]+)/);
            const funcName = funcMatch ? funcMatch[1] : null;
            inFunctionToComment = funcName ? testsToRemove.has(funcName) : false;
        } else if (line.length > 0 && !/^\s/.test(line)) {
            inFunctionToComment = false;
        }
        
        if (inFunctionToComment) {
            newLines.push(`# ${line}`);
        } else {
            newLines.push(line);
        }
    }
    const modifiedTestCode = newLines.join('\n');
    
    const remainingTestCases = proposedTestCases.filter(tc => !testsToRemove.has(tc.name.split('[')[0]));

    const executePayload = {
        test_code: modifiedTestCode,
        custom_test_code: customTestCode, // <-- ADDED: Also send on rerun
        plugin_repo_code: generatedCode.plugin_repo,
        plugin_tenant_code: generatedCode.plugin_tenant,
        logic_summary: currentLogicPlan,
        proposed_test_cases: remainingTestCases,
    };

    const executeResponse = await callBackend(`${API_BASE_URL}/execute`, executePayload);
    
    if (executeResponse.error) {
        setStatus(3, 'failed', executeResponse.error);
        return;
    }

    generatedCode.plugin_repo = executeResponse.plugin_repo;
    generatedCode.plugin_tenant = executeResponse.plugin_tenant;
    generatedCode.test = modifiedTestCode;
    
    displayFinalReport(executeResponse);
}

async function handleForceSave() {
    const saveBtn = getEl('force-save-btn');
    saveBtn.disabled = true;
    saveBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Saving...';
    const baseFileName = fileNameDisplay.textContent.replace(/\.xlsx?$/, '');
    const defaultPluginName = baseFileName ? `DP0xx${baseFileName}` : 'DP0xxNewPlugin_ForceSaved';

    const payload = {
        plugin_name: extractedPluginName || defaultPluginName,
        repo_code: generatedCode.plugin_repo,
        tenant_code: generatedCode.plugin_tenant,
        test_code: generatedCode.test
    };

    const response = await callBackend(`${API_BASE_URL}/save_plugin`, payload);
    const actionsContainer = getEl('actions-container');

    if (response.error) {
        alert(`Failed to save files: ${response.error}`);
        saveBtn.disabled = false;
        saveBtn.innerHTML = '<i class="fas fa-save mr-2"></i>Ignore Errors & Save Files';
    } else {
        if (actionsContainer) {
            showGitPushUI(actionsContainer);
        }
    }
}

async function handlePushToGit() {
    const pushBtn = getEl('push-to-git-btn');
    const feedbackEl = getEl('git-push-feedback');

    if (!pushBtn || !feedbackEl) return;

    pushBtn.disabled = true;
    pushBtn.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Pushing to Git...';
    feedbackEl.innerHTML = '';
    const baseFileName = fileNameDisplay.textContent.replace(/\.xlsx?$/, '');
    const defaultPluginName = baseFileName ? `DP0xx${baseFileName}` : 'DP0xxNewPlugin_ForceSaved';

    const payload = {
        plugin_name: extractedPluginName || defaultPluginName,
        plugin_repo_code: generatedCode.plugin_repo,
        plugin_tenant_code: generatedCode.plugin_tenant,
        test_code: generatedCode.test
    };

    const response = await callBackend(`${API_BASE_URL}/push_to_git`, payload);

    if (response.error) {
        feedbackEl.innerHTML = `<div class="text-red-700 text-left p-4 bg-red-100 rounded-lg">
            <strong>Git Push Failed:</strong> ${escapeHtml(response.error)}
        </div>`;
        pushBtn.disabled = false;
        pushBtn.innerHTML = '<i class="fas fa-sync-alt mr-2"></i>Retry Push';
    } else {
        feedbackEl.innerHTML = `<div class="text-green-700 text-left p-4 bg-green-100 rounded-lg">
            <h4 class="font-bold"><i class="fas fa-check-circle mr-2"></i>Success!</h4>
            <p>Code pushed to branch: <strong>${escapeHtml(response.branch)}</strong></p>
            ${response.commit_message ? `<pre class="code-block mt-4 text-xs !bg-green-50 !text-green-900">${escapeHtml(response.commit_message)}</pre>` : ''}
        </div>`;
        pushBtn.innerHTML = '<i class="fas fa-check mr-2"></i>Pushed Successfully';
    }
}
    

function showGitPushUI(containerElement) {
    containerElement.innerHTML = `
        <div id="success-actions-container" class="mt-8 pt-6 border-t border-dashed text-left">
            <h4 class="font-semibold text-xl text-gray-800 mb-2">Final Step: Push to Repository</h4>
            <p class="text-gray-600 mb-4">You can now push the final files to the Git repository.</p>
            <div>
                <div class="flex justify-end">
                <button id="push-to-git-btn" class="bg-gray-800 hover:bg-black text-white font-bold py-2 px-4 rounded-lg transition w-full">
                    <i class="fab fa-git-alt mr-2"></i>Push Files to Git Branch
                </button>
                </div>
                <div id="git-push-feedback" class="mt-4 text-sm font-medium"></div>
            </div>
        </div>
    `;
    getEl('push-to-git-btn')?.addEventListener('click', handlePushToGit);
}

function displayFinalReport(response) {
    const { success, output, saved_path, attempt_details, plugin_repo, plugin_tenant } = response;
    
    setStatus(3, success ? 'success' : 'failed');

    const last_attempt = attempt_details[attempt_details.length - 1];
    const passed = last_attempt.passed.length;
    const failed = last_attempt.failed.length;
    const total = passed + failed;

    const summaryColor = success ? 'bg-green-100 border-green-500' : 'bg-red-100 border-red-500';
    const summaryIcon = success ? '<i class="fas fa-check-circle text-green-600 mr-3"></i>' : '<i class="fas fa-times-circle text-red-600 mr-3"></i>';
    const summaryText = success ? 'All Tests Passed!' : `Tests Failed After ${attempt_details.length} Attempt(s).`;

    const exportButtonHTML = `
        <button id="export-csv-btn" class="text-sm bg-white border border-gray-300 hover:bg-gray-100 text-gray-700 font-semibold py-1 px-3 rounded-lg transition-colors ml-auto">
            <i class="fas fa-file-csv mr-2"></i>Export Results to CSV
        </button>
    `;

    const attemptsHTML = attempt_details.map((attempt, index) => {
        const attemptSuccess = attempt.failed.length === 0;
        const attemptIcon = attemptSuccess ? '<i class="fas fa-check-circle text-green-500"></i>' : '<i class="fas fa-times-circle text-red-500"></i>';
        
        let passedHTML = '';
        if (attempt.passed.length > 0) {
            passedHTML = `
                <div class="mt-4">
                    <h5 class="font-semibold text-green-800 mb-2 flex items-center"><i class="fas fa-check mr-2"></i> ${attempt.passed.length} Passed - What was tested:</h5>
                    <ul class="list-disc ml-5 text-sm text-gray-700 space-y-1 bg-green-50 p-3 rounded-md border border-green-200">
                        ${attempt.passed.map(t => `<li><strong class="font-mono text-gray-800">${t.name}</strong>: ${t.description}</li>`).join('')}
                    </ul>
                </div>
            `;
        }

        let failedHTML = '';
        if (attempt.failed.length > 0) {
            failedHTML = `
                <div class="mt-4">
                    <h5 class="font-semibold text-red-800 mb-2 flex items-center"><i class="fas fa-times mr-2"></i> ${attempt.failed.length} Failed - What was tested:</h5>
                    <ul class="list-disc ml-5 text-sm text-gray-700 space-y-2 bg-red-50 p-3 rounded-md border border-red-200">
                        ${attempt.failed.map(t => `<li>
                            <strong class="font-mono text-gray-800">${t.name}</strong>: ${t.description}
                            <pre class="text-xs text-red-700 mt-2 p-2 bg-red-100 rounded-md whitespace-pre-wrap font-mono"><strong>Reason:</strong>\n${escapeHtml(t.reason)}</pre>
                        </li>`).join('')}
                    </ul>
                </div>
            `;
        }
        
        let whatsNext = "";
        const isLastAttempt = index === attempt_details.length - 1;
        if (!attemptSuccess && !isLastAttempt) {
            whatsNext = `<div class="mt-4 text-sm text-indigo-700 bg-indigo-50 p-2 rounded-md"><i class="fas fa-magic mr-2"></i><strong>Debugger Agent:</strong> Analyzing failures and attempting a fix for the next run...</div>`;
        } else if (attemptSuccess) {
            whatsNext = `<div class="mt-4 text-sm text-green-700 bg-green-50 p-2 rounded-md"><i class="fas fa-check-circle mr-2"></i><strong>Success:</strong> All tests passed in this attempt!</div>`;
        } else {
            whatsNext = `<div class="mt-4 text-sm text-red-700 bg-red-50 p-2 rounded-md"><i class="fas fa-exclamation-triangle mr-2"></i><strong>Final Result:</strong> Debugger reached max attempts and could not fix all issues.</div>`;
        }

        return `
            <details class="bg-white rounded-lg shadow mt-4" ${index === attempt_details.length - 1 ? 'open' : ''}>
                <summary class="cursor-pointer p-4 font-semibold text-gray-700 hover:bg-gray-50 flex justify-between items-center">
                    <span class="flex items-center gap-3">
                        ${attemptIcon}
                        Attempt ${attempt.attempt}: ${attempt.passed.length} Passed, ${attempt.failed.length} Failed
                    </span>
                    <i class="fas fa-chevron-down transition-transform"></i>
                </summary>
                <div class="p-4 border-t">
                    ${failedHTML}
                    ${passedHTML}
                    ${whatsNext}
                </div>
            </details>
        `;
    }).join('');
    
    const finalCodeSectionHTML = `
        <h4 class="font-semibold text-xl text-gray-800 mb-4 mt-8"><i class="fas fa-code mr-2 text-indigo-500"></i>Final Generated Code</h4>
        <div class="grid md:grid-cols-2 gap-6">
            <div>
                <h5 class="font-semibold text-md text-gray-800 mb-2">Tenant File</h5>
                <div class="code-block"><button class="copy-btn" onclick="copyCode(this)"><i class="fas fa-copy"></i></button><code class="language-python">${escapeHtml(plugin_tenant)}</code></div>
            </div>
            <div>
                <h5 class="font-semibold text-md text-gray-800 mb-2">Repo File</h5>
                <div class="code-block"><button class="copy-btn" onclick="copyCode(this)"><i class="fas fa-copy"></i></button><code class="language-python">${escapeHtml(plugin_repo)}</code></div>
            </div>
        </div>`;
    
    let finalSectionHTML = '';
    if (success) {
        finalSectionHTML = `
            ${saved_path ? `<div class="mt-6 p-4 bg-blue-50 border-blue-200 border rounded-lg text-center text-blue-800">✅ Plugin successfully created and files saved to folder: <strong>${saved_path}</strong></div>` : ''}

            <details class="bg-gray-100 rounded-lg mt-6">
                <summary class="cursor-pointer p-4 font-semibold text-gray-800 hover:bg-gray-200">View Full Raw Execution & Debug Log <i class="fas fa-chevron-down ml-2 text-sm"></i></summary>
                <div class="p-4 border-t border-gray-300"><pre class="code-block" style="max-height: 500px;">${escapeHtml(output)}</pre></div>
            </details>

            <div id="success-actions-container" class="mt-8 pt-6 border-t border-dashed text-left">
                <h4 class="font-semibold text-xl text-gray-800 mb-2">Final Step: Push to Repository</h4>
                <p class="text-gray-600 mb-4">You can now push the final files to the Git repository.</p>
                <div>
                    <div class="flex justify-end">
                    <button id="push-to-git-btn" class="bg-gray-800 hover:bg-black text-white font-bold py-2 px-4 rounded-lg transition w-full">
                        <i class="fab fa-git-alt mr-2"></i>Push Files to Git Branch
                    </button>
                    </div>
                    <div id="git-push-feedback" class="mt-4 text-sm font-medium"></div>
                </div>
            </div>
        `;
    } else {
        const failedTestItems = last_attempt.failed.map(t => {
            const testName = escapeHtml(t.name);
            const isRemoved = testsToExclude.has(testName.split('[')[0]);
            return `
            <div class="failed-test-item border rounded-lg p-3 flex items-center justify-between bg-red-50" style="opacity: ${isRemoved ? '0.4' : '1'}">
                <div><strong class="font-mono text-gray-800">${testName}</strong></div>
                <button data-test-name="${testName}" class="remove-test-btn text-sm bg-white border border-gray-300 hover:bg-gray-100 text-gray-700 font-semibold py-1 px-3 rounded-lg transition-colors" ${isRemoved ? 'disabled' : ''}>
                    ${isRemoved ? '<i class="fas fa-check mr-2"></i>Removed' : '<i class="fas fa-trash-alt mr-2"></i>Remove'}
                </button>
            </div>`
        }).join('');

        const actionsHTML = `
            <div id="rerun-container" class="mt-8 pt-6 border-t border-dashed">
                <h4 class="font-semibold text-xl text-yellow-800 mb-2 flex items-center"><i class="fas fa-wrench mr-3"></i>Debugger Actions</h4>
                <p class="text-gray-600 mb-4">You can remove specific failing tests and re-run the debugger, or accept the current code and save the files.</p>
                <div id="failed-test-list" class="space-y-2 mb-4">${failedTestItems}</div>
                <div class="flex items-center space-x-4">
                    <button id="rerun-debugger-btn" class="bg-yellow-500 hover:bg-yellow-600 text-white font-bold py-2 px-4 rounded-lg transition"><i class="fas fa-redo mr-2"></i>Re-run with Selected Tests</button>
                    <button id="force-save-btn" class="bg-green-600 hover:bg-green-700 text-white font-bold py-2 px-4 rounded-lg transition"><i class="fas fa-save mr-2"></i>Ignore Errors & Save Files</button>
                </div>
            </div>`;
            
        finalSectionHTML = `
            <div id="actions-container">${actionsHTML}</div>
            <details class="bg-gray-100 rounded-lg mt-6">
                <summary class="cursor-pointer p-4 font-semibold text-gray-800 hover:bg-gray-200">View Full Raw Execution & Debug Log <i class="fas fa-chevron-down ml-2 text-sm"></i></summary>
                <div class="p-4 border-t border-gray-300"><pre class="code-block" style="max-height: 500px;">${escapeHtml(output)}</pre></div>
            </details>
        `;
    }

    contents[3].innerHTML = `
        <div class="p-4 rounded-lg border-l-4 ${summaryColor} mb-6">
            <div class="flex items-center">
                <h3 class="font-bold text-lg flex items-center">${summaryIcon}${summaryText}</h3>
                ${exportButtonHTML}
            </div>
            <div class="flex space-x-6 mt-2">
                <div class="text-green-700"><strong>${passed}</strong> Passed</div>
                <div class="text-red-700"><strong>${failed}</strong> Failed</div>
                <div><strong>${total}</strong> Total</div>
            </div>
        </div>
        <h4 class="font-semibold text-xl text-gray-800 mb-2 mt-6">Debugger Agent Report</h4>
        <p class="text-gray-600 mb-2">The following is a breakdown of each automated test and debug attempt.</p>
        <div id="attempts-container">${attemptsHTML}</div>
        ${finalCodeSectionHTML}
        ${finalSectionHTML}
    `;

    getEl('export-csv-btn')?.addEventListener('click', () => {
        const pluginName = extractedPluginName || 'GeneratedPlugin';
        exportTestResultsToCSV(response.attempt_details, pluginName);
    });

    if (success) {
        getEl('push-to-git-btn')?.addEventListener('click', handlePushToGit);
    } else {
        document.querySelectorAll('.remove-test-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const testName = e.currentTarget.dataset.testName;
                const baseTestName = testName.split('[')[0];
                testsToExclude.add(baseTestName);
                const testItem = e.currentTarget.closest('.failed-test-item');
                testItem.style.opacity = '0.4';
                e.currentTarget.disabled = true;
                e.currentTarget.innerHTML = '<i class="fas fa-check mr-2"></i>Removed';
            });
        });
        getEl('rerun-debugger-btn')?.addEventListener('click', handleRerunDebugger);
        getEl('force-save-btn')?.addEventListener('click', handleForceSave);
    }
}

function copyCode(button) {
    const textToCopy = button.nextElementSibling.textContent;
    navigator.clipboard.writeText(textToCopy).then(() => {
        const originalIcon = button.innerHTML;
        button.innerHTML = '<i class="fas fa-check"></i>';
        setTimeout(() => { button.innerHTML = originalIcon; }, 1500);
    }).catch(err => console.error('Failed to copy text: ', err));
}

function escapeHtml(unsafe) {
    if (!unsafe) return '';
    return unsafe
         .replace(/&/g, "&amp;")
         .replace(/</g, "&lt;")
         .replace(/>/g, "&gt;")
         .replace(/"/g, "&quot;")
         .replace(/'/g, "&#039;");
}

function resetSession() {
    location.reload();
}

fileUpload.addEventListener('change', () => {
    if (fileUpload.files.length > 0) {
        fileNameDisplay.textContent = fileUpload.files[0].name;
        validateBtn.classList.remove('hidden');
    }
});
validateBtn.addEventListener('click', handleValidation);
resetBtn.addEventListener('click', resetSession);

setStatus(1, 'pending'); setStatus('1a', 'pending'); setStatus(2, 'pending'); setStatus(3, 'pending');