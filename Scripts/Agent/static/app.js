    const state = { actions: [], tasks: [], current: null, form: null, jobs: [], runningJobs: [], progressTimers: {}, ai3dTimers: {}, lang: "en" };
    const uiText = {
      en: {
        brand: "Smart Assistant",
        heroTitle: "Hello, welcome to Completed Script Agent!",
        heroIntro: "Describe what you want to do, and the system will complete the parameters on this page before opening the required terminal to run the script.",
        heroFlow: "Common workflows: build environment, register task, collect data, augment data, convert LeRobot, train model, and run inference.",
        inputPlaceholder: "Enter your question, for example: I want to collect data",
        sendTitle: "Send",
        stopTitle: "Stop current task",
        runButton: "Start",
        writeTaskButton: "Write Task",
        clearButton: "Clear",
        userPrefix: "You",
        agentPrefix: "Agent",
        errorPrefix: "Error",
        launchFailurePrefix: "Launch failed",
        recognizedPrefix: "Recognized as",
        recognizedSuffix: "Please fill in the parameters.",
        completed: "Completed",
        taskId: "Task ID",
        dataset: "Dataset",
      },
      zh: {
        brand: "智能问询",
        heroTitle: "你好，欢迎来到 Completed Script Agent！",
        heroIntro: "你可以直接描述要做的事，系统会在网页中补齐参数，再打开需要的终端执行脚本。",
        heroFlow: "常用流程：搭建环境、注册任务、采集数据、数据增强、转换 LeRobot、训练模型、运行推理。",
        inputPlaceholder: "请输入您的问题，例如：我要采集数据",
        sendTitle: "发送",
        stopTitle: "停止当前任务",
        runButton: "启动",
        writeTaskButton: "写入任务",
        clearButton: "清空",
        userPrefix: "你",
        agentPrefix: "Agent",
        errorPrefix: "出错",
        launchFailurePrefix: "启动失败",
        recognizedPrefix: "已识别为",
        recognizedSuffix: "请补充参数。",
        completed: "已完成",
        taskId: "任务 ID",
        dataset: "数据集",
      },
    };
    const quick = [
      [{en: "Build Environment", zh: "搭建环境"}, {en: "Open the default USD scene, then save changes for task registration.", zh: "打开默认 USD 场景，修改后另存用于注册任务。"}, "build_environment"],
      [{en: "Analyze Assets", zh: "分析资产"}, {en: "Analyze required, available, and missing USD assets for the lab.", zh: "分析实验室需要哪些 USD 资产，以及已有和缺失资产。"}, "analyze_assets"],
      [{en: "Register Task", zh: "注册任务"}, {en: "Fill in the USD path and task description, then write to task_registry.", zh: "填写 USD 路径和任务描述，写入 task_registry。"}, "register_task"],
      [{en: "Collect Data", zh: "采集数据"}, {en: "Choose a task and episode count, then open the collection terminal.", zh: "选择任务和条数，打开采集终端。"}, "collect_data"],
      [{en: "Augment Data", zh: "数据增强"}, {en: "Apply lighting, speed, and camera perturbation augmentation to HDF5 data.", zh: "对 HDF5 数据做光照、速度和相机扰动增强。"}, "augment_data"],
      [{en: "Convert LeRobot", zh: "转换 LeRobot"}, {en: "Convert HDF5 data into a LeRobot dataset.", zh: "把 HDF5 转成 LeRobot 数据集。"}, "convert_lerobot"],
      [{en: "Train Model", zh: "训练模型"}, {en: "Start the LeRobot training command.", zh: "启动 LeRobot 训练命令。"}, "train_model"],
      [{en: "Run Inference", zh: "运行推理"}, {en: "Start the model service and inference client automatically.", zh: "自动启动模型服务和推理客户端。"}, "run_inference"],
    ];
    const stepNames = {
      en: ["Identify Intent", "Fill Parameters", "Launch Terminal", "Result"],
      zh: ["识别意图", "补充参数", "启动终端", "任务结果"],
    };
    const actionText = {
      collect_data: {
        en: ["Collect Data", "Choose a task and collection count, then launch the IsaacLab keyboard teleoperation terminal."],
        zh: ["采集数据", "选择任务和采集条数，启动 IsaacLab 键盘遥操作采集终端。"],
      },
      build_environment: {
        en: ["Build Environment", "Open the preset USD scene in Isaac Sim, save the modified scene, then use it for task registration."],
        zh: ["搭建环境", "打开预设 USD 场景，在 Isaac Sim 中修改并另存，之后用于注册任务。"],
      },
      analyze_assets: {
        en: ["Analyze Assets", "Describe the lab or task scene. The system scans the asset folder and analyzes available and missing assets."],
        zh: ["分析资产", "输入要搭建的实验室或任务场景描述，系统会扫描 asset 文件夹并分析已有资产和缺失资产。"],
      },
      register_task: {
        en: ["Register Task", "Enter the saved USD path and task description. The system generates a task_id and writes it to task_registry."],
        zh: ["注册任务", "输入保存后的 USD 路径和任务描述，系统生成 task_id 并写入 task_registry。"],
      },
      convert_lerobot: {
        en: ["Convert LeRobot", "Choose a task and HDF5 path. Other parameters use the recommended defaults."],
        zh: ["转换 LeRobot", "选择任务和 HDF5 路径，其余参数使用推荐默认值。"],
      },
      train_model: {
        en: ["Train Model", "Train ACT, SmolVLA, or PI0 with the LeRobot platform."],
        zh: ["训练模型", "使用 LeRobot 平台训练 ACT、SmolVLA 或 PI0。"],
      },
      run_inference: {
        en: ["Inference / Evaluation", "Start the model service first, then launch the IsaacSim inference client."],
        zh: ["推理/评估", "先自动启动模型服务，再启动 IsaacSim 推理客户端。"],
      },
      augment_data: {
        en: ["Augment Data", "Apply lighting, temporal speed, camera jitter, and other augmentation to existing HDF5 data."],
        zh: ["增强数据", "对已有 HDF5 数据做光照、时间速度、相机扰动等增强。"],
      },
      replay_data: {
        en: ["Replay Data", "Replay HDF5 trajectories to inspect collection results."],
        zh: ["回放数据", "回放 HDF5 轨迹，检查采集结果。"],
      },
    };
    const fieldText = {
      task_id: {en: "Task", zh: "任务"},
      dataset_file: {en: "Dataset Path", zh: "数据保存路径"},
      num_demos: {en: "Demo Count", zh: "采集 demo 条数"},
      gui: {en: "Run with GUI", zh: "有界面运行"},
      usd_path: {en: "USD Scene Path", zh: "USD 场景路径"},
      asset_dir: {en: "Asset Folder", zh: "资源文件夹"},
      lab_description: {en: "Lab / Scene Description", zh: "实验室/场景描述"},
      description: {en: "Task Description", zh: "任务描述"},
      task_optional: {en: "Use Task Preset", zh: "使用任务预设"},
      hdf5_path: {en: "HDF5 Input Path", zh: "HDF5 输入路径"},
      output_dir: {en: "Output Directory", zh: "模型输出目录"},
      policy_type: {en: "Model / Client Type", zh: "模型/客户端类型"},
      dataset_repo_id: {en: "LeRobot Dataset Directory (absolute path)", zh: "LeRobot 数据集目录（绝对路径）"},
      batch_size: {en: "Batch Size", zh: "batch size"},
      steps: {en: "Training Steps", zh: "训练步数 steps"},
      pi0_freeze_vision_encoder: {en: "PI0: Freeze Vision Encoder", zh: "PI0: freeze vision encoder"},
      pi0_train_expert_only: {en: "PI0: Train Expert Only", zh: "PI0: train expert only"},
      pi0_gradient_checkpointing: {en: "PI0: Gradient Checkpointing", zh: "PI0: gradient checkpointing"},
      wandb_enable: {en: "Enable wandb", zh: "启用 wandb"},
      wandb_project: {en: "wandb Project", zh: "wandb project"},
      wandb_entity: {en: "wandb Entity / Account", zh: "wandb entity / 账号"},
      wandb_notes: {en: "wandb Notes", zh: "wandb notes"},
      push_to_hub: {en: "Upload to Hugging Face Hub", zh: "上传到 Hugging Face Hub"},
      hub_repo_id: {en: "Hugging Face Repo ID (account/model)", zh: "Hugging Face repo id（账号/模型名）"},
      hub_private: {en: "Private Hub Repository", zh: "Hub 私有仓库"},
      policy_path: {en: "Model Path or Hugging Face Repo ID", zh: "模型路径或 Hugging Face repo id"},
      episodes: {en: "Evaluation Episodes", zh: "评估回合数"},
      device: {en: "Device", zh: "运行设备"},
      strict: {en: "Strict Model Config Loading", zh: "严格加载模型配置"},
      output_file: {en: "Output File Path (optional)", zh: "输出文件路径（可空）"},
      worker_count: {en: "Parallel Worker Count", zh: "并行 worker 数量"},
      light_scales: {en: "Lighting Augmentation Scale", zh: "光照增强倍率"},
      temporal_scales: {en: "Temporal Speed Scale", zh: "时间速度倍率"},
      camera_jitter_count: {en: "Camera Jitter Count", zh: "相机扰动数量"},
      include_original: {en: "Include Original Demos", zh: "包含原始 demo"},
      demo_index: {en: "Demo Index (-1 for all)", zh: "demo 编号，-1 表示全部"},
      replay_mode: {en: "Replay Mode", zh: "回放模式"},
    };
    const jobTitleText = {
      "模型服务": {en: "Model Service", zh: "模型服务"},
      "运行推理": {en: "Run Inference", zh: "运行推理"},
      "采集数据": {en: "Collect Data", zh: "采集数据"},
      "搭建环境": {en: "Build Environment", zh: "搭建环境"},
      "训练模型": {en: "Train Model", zh: "训练模型"},
      "转换 LeRobot": {en: "Convert LeRobot", zh: "转换 LeRobot"},
      "增强数据": {en: "Augment Data", zh: "增强数据"},
    };

    function el(tag, attrs={}, children=[]) {
      const node = document.createElement(tag);
      for (const [k,v] of Object.entries(attrs)) {
        if (k === "class") node.className = v;
        else if (k === "text") node.textContent = v;
        else if (k === "html") node.innerHTML = v;
        else node.setAttribute(k, v);
      }
      for (const child of children) node.appendChild(child);
      return node;
    }
    function setLanguage(lang) {
      state.lang = lang;
      document.documentElement.lang = lang === "zh" ? "zh-CN" : "en";
      const copy = uiText[lang] || uiText.en;
      document.querySelectorAll("[data-i18n]").forEach(node => {
        const key = node.getAttribute("data-i18n");
        if (copy[key]) node.textContent = copy[key];
      });
      const input = document.getElementById("chatInput");
      if (input) input.placeholder = copy.inputPlaceholder;
      const toggle = document.getElementById("langToggle");
      if (toggle) {
        const useChinese = lang === "en";
        toggle.textContent = useChinese ? "中文" : "EN";
        toggle.title = useChinese ? "切换到中文" : "Switch to English";
        toggle.setAttribute("aria-label", toggle.title);
      }
      renderNav();
      updateSendButton();
    }
    function currentCopy() {
      return uiText[state.lang] || uiText.en;
    }
    function localizedAction(action, form) {
      const copy = actionText[action]?.[state.lang] || actionText[action]?.en;
      return {
        title: copy?.[0] || form.title || "",
        description: copy?.[1] || form.description || "",
      };
    }
    function localizedFieldLabel(field) {
      return fieldText[field.name]?.[state.lang] || fieldText[field.name]?.en || field.label || "";
    }
    function localizedForm(form, action) {
      const copy = localizedAction(action, form);
      return {
        ...form,
        title: copy.title,
        description: copy.description,
        fields: (form.fields || []).map(field => ({
          ...field,
          label: localizedFieldLabel(field),
        })),
      };
    }
    function translatePlainText(text) {
      const value = String(text ?? "");
      if (state.lang === "zh") return value;
      if (jobTitleText[value]) return jobTitleText[value].en;
      let match = value.match(/^已启动模型服务和推理客户端：(.+)$/);
      if (match) return `Started model service and inference client: ${match[1]}`;
      match = value.match(/^模型服务已启动，但 (\d+) 秒内未检测到服务端打印 cuda，暂未启动推理客户端。请查看模型服务终端输出。$/);
      if (match) return `Model service started, but no server cuda log was detected within ${match[1]} seconds. The inference client was not started. Check the model service terminal output.`;
      match = value.match(/^已尝试停止最近启动的一组任务，共 (\d+) 个。$/);
      if (match) return `Attempted to stop the most recently started task group, total ${match[1]}.`;
      match = value.match(/^已尝试停止 (\d+) 个 Web Agent 记录的任务。$/);
      if (match) return `Attempted to stop ${match[1]} task(s) recorded by Web Agent.`;
      match = value.match(/^已启动训练：(.+)$/);
      if (match) return `Started training: ${match[1]}`;
      match = value.match(/^已启动 LeRobot 转换：(.+)$/);
      if (match) return `Started LeRobot conversion: ${match[1]}`;
      match = value.match(/^已启动数据增强：(.+)$/);
      if (match) return `Started data augmentation: ${match[1]}`;
      match = value.match(/^已启动采集终端：(.+)，目标 (\d+) 条 demo。$/);
      if (match) return `Started data collection terminal: ${match[1]}, target ${match[2]} demos.`;
      const fixed = {
        "已完成": "Completed",
        "已完成资产分析。": "Asset analysis completed.",
        "已打开搭建环境终端。请在 Isaac Sim 中修改并另存 USD。": "Opened the environment setup terminal. Modify the scene in Isaac Sim and save the USD.",
      };
      return fixed[value] || value;
    }
    function setSteps(active=0, done=[]) {
      const box = document.getElementById("steps");
      if (!box) return;
      box.innerHTML = "";
      (stepNames[state.lang] || stepNames.en).forEach((name, i) => {
        const cls = done.includes(i) ? "step done" : i === active ? "step active" : "step";
        box.appendChild(el("div", {class: cls}, [
          el("div", {class: "step-badge", text: done.includes(i) ? "✓" : String(i+1)}),
          el("div", {}, [el("div", {class: "step-name", text: name}), el("div", {class: "mini", text: state.lang === "zh" ? (done.includes(i) ? "已完成" : i === active ? "进行中" : "未开始") : (done.includes(i) ? "Done" : i === active ? "In progress" : "Not started")})])
        ]));
      });
    }
    function updateSendButton() {
      const btn = document.getElementById("sendBtn");
      const copy = currentCopy();
      if (state.runningJobs.length > 0) {
        btn.title = copy.stopTitle;
        btn.innerHTML = '<span class="stop-square"></span>';
      } else {
        btn.title = copy.sendTitle;
        btn.innerHTML = '<span class="send-arrow">↑</span>';
      }
    }
    function clearProgressTimers() {
      Object.values(state.progressTimers).forEach(timer => clearInterval(timer));
      state.progressTimers = {};
    }
    function renderNav() {
      const nav = document.getElementById("navList");
      nav.innerHTML = "";
      quick.forEach(([title,, action]) => {
        const item = el("div", {class: "nav-item" + (state.current === action ? " active" : ""), text: title[state.lang] || title.en});
        item.onclick = () => loadForm(action, {});
        nav.appendChild(item);
      });
    }
    function renderMessage(html, options={}) {
      const ws = document.getElementById("workspace");
      const attrs = {class: "message" + (options.clearable ? " run-output" : ""), html};
      if (options.action) attrs["data-output-action"] = options.action;
      ws.insertBefore(el("div", attrs), ws.firstChild);
    }
    function clearCurrentRunOutput() {
      clearProgressTimers();
      document.querySelectorAll(".run-output, .loading-panel").forEach(node => {
        const outputAction = node.getAttribute("data-output-action") || node.getAttribute("data-loading") || "";
        if (!state.current || !outputAction || outputAction === state.current) node.remove();
      });
    }
    function actionLoadingText(action) {
      const map = {
        analyze_assets: {zh: ["正在分析资产", "正在扫描 USD 资产库，并调用模型分析已有资产与缺失资产"], en: ["Analyzing assets", "Scanning the USD asset library and analyzing available and missing assets"]},
        register_task: {zh: ["正在注册任务", "正在生成 task_id 和语言指令，并写入 task_registry"], en: ["Registering task", "Generating task_id and language instructions, then writing to task_registry"]},
        build_environment: {zh: ["正在打开环境", "正在准备 Isaac Sim 场景编辑终端"], en: ["Opening environment", "Preparing the Isaac Sim scene editing terminal"]},
        collect_data: {zh: ["正在启动采集", "正在创建采集终端并准备 IsaacLab"], en: ["Starting collection", "Creating the collection terminal and preparing IsaacLab"]},
        augment_data: {zh: ["正在启动增强", "正在创建数据增强终端"], en: ["Starting augmentation", "Creating the data augmentation terminal"]},
        convert_lerobot: {zh: ["正在转换准备", "正在创建 LeRobot 转换终端"], en: ["Preparing conversion", "Creating the LeRobot conversion terminal"]},
        train_model: {zh: ["正在启动训练", "正在创建 LeRobot 训练终端"], en: ["Starting training", "Creating the LeRobot training terminal"]},
        run_inference: {zh: ["正在启动推理", "正在启动模型服务并准备推理客户端"], en: ["Starting inference", "Starting the model service and preparing the inference client"]},
      };
      return map[action]?.[state.lang] || map[action]?.en || (state.lang === "zh" ? ["正在处理", "Agent 正在执行当前请求"] : ["Processing", "Agent is running the current request"]);
    }
    function showLoading(action) {
      const formPanel = document.querySelector(".form-panel");
      const [title, detail] = actionLoadingText(action);
      const panel = el("div", {class: "loading-panel run-output", "data-loading": action, "data-output-action": action}, [
        el("div", {class: "loading-spinner"}),
        el("div", {}, [
          el("strong", {class: "loading-dots", text: title}),
          el("div", {text: state.lang === "zh" ? detail + "，请稍等。" : detail + ". Please wait."})
        ])
      ]);
      if (formPanel) formPanel.appendChild(panel);
      else document.getElementById("workspace").appendChild(panel);
      panel.scrollIntoView({behavior: "smooth", block: "center"});
      return panel;
    }
    function setFormBusy(form, busy) {
      if (!form) return;
      form.fields.forEach(f => {
        const input = document.querySelector(`[name="${f.name}"]`);
        if (input) input.disabled = busy;
      });
      document.querySelectorAll(".actions button").forEach(button => {
        button.disabled = busy;
      });
    }
    function escapeHtml(text) {
      return String(text).replace(/[&<>"']/g, ch => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }[ch]));
    }
    function selectedTask() {
      const taskInput = document.querySelector(`[name="task_id"]`);
      if (!taskInput) return null;
      return (state.tasks || []).find(task => String(task.task_id) === String(taskInput.value)) || null;
    }
    function syncTaskDatasetField(form) {
      if (!form) return;
      const task = selectedTask();
      if (!task || !task.dataset_file) return;
      const fieldNames = new Set((form.fields || []).map(f => f.name));
      const datasetInput = document.querySelector(`[name="dataset_file"]`);
      if (datasetInput && fieldNames.has("dataset_file")) datasetInput.value = task.dataset_file;
      const hdf5Input = document.querySelector(`[name="hdf5_path"]`);
      const useTaskInput = document.querySelector(`[name="task_optional"]`);
      if (hdf5Input && fieldNames.has("hdf5_path") && (!useTaskInput || useTaskInput.checked)) {
        hdf5Input.value = task.dataset_file;
      }
    }
    function renderForm(form, action) {
      const viewForm = localizedForm(form, action);
      state.form = viewForm; state.current = action; renderNav(); setSteps(1, [0]);
      const ws = document.getElementById("workspace");
      ws.innerHTML = "";
      const panel = el("div", {class: "form-panel"});
      panel.appendChild(el("h2", {text: viewForm.title}));
      panel.appendChild(el("p", {text: viewForm.description || ""}));
      const grid = el("div", {class: "form-grid"});
      viewForm.fields.forEach(f => grid.appendChild(renderField(f)));
      panel.appendChild(grid);
      const actions = el("div", {class: "actions"});
      const copy = currentCopy();
      const run = el("button", {type: "button", text: action === "register_task" ? copy.writeTaskButton : copy.runButton});
      run.onclick = () => submitAction(action, viewForm);
      actions.appendChild(run);
      actions.appendChild(el("button", {type: "button", class: "secondary", text: copy.clearButton}, []));
      actions.lastChild.onclick = () => { clearCurrentRunOutput(); };
      panel.appendChild(actions);
      if (action === "build_environment") panel.appendChild(renderHunyuan3DPanel());
      ws.appendChild(panel);
      updateConditionalFields(viewForm);
      if (action === "build_environment") loadHunyuan3DAssets();
    }
    function renderField(f) {
      const label = el("label", {"data-field": f.name});
      if (f.show_if) label.setAttribute("data-show-if", f.show_if);
      if (f.show_value !== undefined) label.setAttribute("data-show-value", String(f.show_value));
      label.appendChild(el("span", {text: f.label}));
      if (f.type === "select") {
        const input = el("select", {name: f.name});
        (f.options || []).forEach(o => {
          const opt = el("option", {value: o.value, text: o.label});
          if (String(o.value) === String(f.default)) opt.selected = true;
          input.appendChild(opt);
        });
        input.addEventListener("change", () => {
          updateConditionalFields(state.form);
          if (f.name === "task_id") syncTaskDatasetField(state.form);
        });
        label.appendChild(input);
      } else if (f.type === "textarea") {
        label.appendChild(el("textarea", {name: f.name}, [])).value = f.default || "";
      } else if (f.type === "checkbox") {
        label.className = "switch-row";
        const input = el("input", {type: "checkbox", name: f.name});
        input.checked = !!f.default;
        input.addEventListener("change", () => updateConditionalFields(state.form));
        label.innerHTML = "";
        label.appendChild(input);
        label.appendChild(el("span", {class: "switch"}));
        label.appendChild(el("span", {text: f.label}));
      } else {
        const attrs = {name: f.name, type: f.type === "number" ? "number" : "text", value: f.default ?? ""};
        if (f.min !== undefined) attrs.min = f.min;
        if (f.max !== undefined) attrs.max = f.max;
        label.appendChild(el("input", attrs));
      }
      return label;
    }
    function updateConditionalFields(form) {
      if (!form) return;
      form.fields.forEach(f => {
        if (!f.show_if) return;
        const wrapper = document.querySelector(`[data-field="${f.name}"]`);
        const controller = document.querySelector(`[name="${f.show_if}"]`);
        let visible = false;
        if (controller && f.show_value !== undefined) {
          visible = String(controller.value) === String(f.show_value);
        } else {
          visible = !!(controller && controller.checked);
        }
        if (!wrapper) return;
        wrapper.style.display = visible ? "" : "none";
        wrapper.querySelectorAll("input, select, textarea").forEach(input => {
          input.disabled = !visible;
        });
      });
    }
    function collectParams(form) {
      const params = {};
      form.fields.forEach(f => {
        const input = document.querySelector(`[name="${f.name}"]`);
        if (!input) return;
        if (input.disabled) return;
        if (f.type === "checkbox") {
          params[f.name] = input.checked;
        } else if (f.type === "number") {
          const value = Number(input.value);
          params[f.name] = Number.isFinite(value) ? value : input.value;
        } else {
          params[f.name] = input.value;
        }
      });
      return params;
    }
    async function api(path, payload) {
      const res = await fetch(path, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload || {})});
      const data = await res.json();
      if (!res.ok || data.ok === false) throw new Error(data.error || data.message || "请求失败");
      return data;
    }
    function renderHunyuan3DPanel() {
      const zh = state.lang === "zh";
      const panel = el("section", {class: "ai3d-panel"});
      panel.appendChild(el("div", {class: "ai3d-heading"}, [
        el("div", {}, [
          el("h3", {text: zh ? "腾讯混元生3D" : "Tencent Hunyuan 3D"}),
          el("p", {text: zh ? "支持文字或本地单图/多视图生成，完成后自动下载为 USDZ。" : "Generate from text or local single/multi-view images and download the result as USDZ."})
        ]),
        el("span", {class: "ai3d-badge", text: "USDZ"})
      ]));
      panel.appendChild(el("div", {class: "ai3d-output"}, [
        el("span", {text: zh ? "保存目录：" : "Output: "}),
        el("code", {id: "ai3dOutputDir", text: zh ? "正在读取..." : "Loading..."})
      ]));

      const grid = el("div", {class: "ai3d-grid"});
      const nameLabel = el("label", {}, [
        el("span", {text: zh ? "3D 资产名称（可选）" : "3D Asset Name (optional)"}),
      ]);
      const assetName = el("input", {id: "ai3dAssetName", type: "text", maxlength: "100"});
      assetName.placeholder = zh ? "例如：白色试管（自动保存为 .usdz）" : "Example: white_test_tube (saved as .usdz)";
      nameLabel.appendChild(assetName);
      nameLabel.appendChild(el("small", {
        class: "field-help",
        text: zh ? "留空时使用系统生成的文件名；同名资产会自动追加序号。" : "Leave blank for an automatic name; duplicates receive a number."
      }));
      grid.appendChild(nameLabel);

      const sourceLabel = el("label", {}, [el("span", {text: zh ? "生成来源" : "Source"})]);
      const source = el("select", {id: "ai3dSource"});
      source.appendChild(el("option", {value: "text", text: zh ? "文字生成3D" : "Text to 3D"}));
      source.appendChild(el("option", {value: "image_file", text: zh ? "本地图片生成3D" : "Local Image to 3D"}));
      sourceLabel.appendChild(source);
      grid.appendChild(sourceLabel);

      const modelLabel = el("label", {}, [el("span", {text: zh ? "模型版本" : "Model Version"})]);
      const model = el("select", {id: "ai3dModel"});
      model.appendChild(el("option", {value: "3.0", text: zh ? "3.0（默认，支持全部模式）" : "3.0 (default, all modes)"}));
      model.appendChild(el("option", {value: "3.1", text: zh ? "3.1（新版，不支持低多边形）" : "3.1 (newer, no Low Poly)"}));
      modelLabel.appendChild(model);
      grid.appendChild(modelLabel);

      const typeLabel = el("label", {}, [el("span", {text: zh ? "生成模式" : "Generation Mode"})]);
      const type = el("select", {id: "ai3dGenerateType"});
      [
        ["Normal", zh ? "标准模型" : "Normal"],
        ["LowPoly", zh ? "低多边形" : "Low Poly"],
        ["Geometry", zh ? "白模" : "Geometry"],
        ["Sketch", zh ? "草图风格" : "Sketch"],
      ].forEach(([value, text]) => type.appendChild(el("option", {value, text})));
      typeLabel.appendChild(type);
      typeLabel.appendChild(el("small", {id: "ai3dModeHelp", class: "field-help"}));
      grid.appendChild(typeLabel);

      const promptLabel = el("label", {id: "ai3dPromptField"}, [
        el("span", {text: zh ? "模型描述" : "Model Prompt"}),
      ]);
      const prompt = el("textarea", {id: "ai3dPrompt"});
      prompt.placeholder = zh ? "例如：实验室桌面上的白色离心机，写实风格，结构清晰" : "Example: a white laboratory centrifuge, realistic and structurally clear";
      promptLabel.appendChild(prompt);
      grid.appendChild(promptLabel);

      const imageFields = el("section", {id: "ai3dImageFields", class: "ai3d-multiview ai3d-hidden"});
      imageFields.appendChild(el("div", {class: "ai3d-multiview-head"}, [
        el("strong", {text: zh ? "本地图片" : "Local Images"}),
        el("small", {
          class: "field-help",
          text: zh ? "点击方框从文件管理器选择 PNG/JPG；所有图片原文件合计不超过 6 MB。" : "Click a field to choose PNG/JPG files; combined original size must not exceed 6 MB."
        })
      ]));
      const viewGrid = el("div", {class: "ai3d-view-grid"});
      const primaryImageLabel = el("label", {}, [
        el("span", {text: zh ? "主图 / 正视图（必选）" : "Primary / Front View (required)"})
      ]);
      const primaryImage = el("input", {id: "ai3dImageFile", type: "file", accept: "image/png,image/jpeg"});
      primaryImageLabel.appendChild(primaryImage);
      viewGrid.appendChild(primaryImageLabel);

      const viewInputs = [
        ["left", zh ? "左视图（可选）" : "Left View (optional)", "3.0"],
        ["right", zh ? "右视图（可选）" : "Right View (optional)", "3.0"],
        ["back", zh ? "后视图（可选）" : "Back View (optional)", "3.0"],
        ["top", zh ? "顶视图（3.1 可选）" : "Top View (3.1 optional)", "3.1"],
        ["bottom", zh ? "底视图（3.1 可选）" : "Bottom View (3.1 optional)", "3.1"],
        ["left_front", zh ? "左前 45°（3.1 可选）" : "Left Front 45° (3.1 optional)", "3.1"],
        ["right_front", zh ? "右前 45°（3.1 可选）" : "Right Front 45° (3.1 optional)", "3.1"],
      ];
      viewInputs.forEach(([viewType, labelText, minModel]) => {
        const label = el("label", {"data-ai3d-view-label": viewType, "data-min-model": minModel}, [
          el("span", {text: labelText})
        ]);
        label.appendChild(el("input", {
          type: "file",
          accept: "image/png,image/jpeg",
          "data-ai3d-view": viewType,
        }));
        viewGrid.appendChild(label);
      });
      imageFields.appendChild(viewGrid);
      grid.appendChild(imageFields);

      const faceLabel = el("label", {}, [
        el("span", {text: zh ? "面数（可选）" : "Face Count (optional)"}),
      ]);
      const face = el("input", {id: "ai3dFaceCount", type: "number", min: "3000", max: "1500000"});
      face.placeholder = zh ? "默认 500000；范围 3000–1500000" : "Default 500000; range 3000–1500000";
      faceLabel.appendChild(face);
      faceLabel.appendChild(el("small", {class: "field-help", text: zh ? "面数越高细节越多，但文件、耗时和仿真负担也更高；低多边形模式会忽略此项。" : "Higher counts add detail and cost; Low Poly ignores this field."}));
      grid.appendChild(faceLabel);

      const pbrLabel = el("label", {class: "ai3d-check"});
      const pbr = el("input", {id: "ai3dPbr", type: "checkbox"});
      pbr.checked = false;
      pbrLabel.appendChild(pbr);
      pbrLabel.appendChild(el("span", {text: zh ? "生成 PBR 材质（官方默认关闭，会增加耗时和积分）" : "Generate PBR materials (off by default; adds time and credits)"}));
      grid.appendChild(pbrLabel);
      panel.appendChild(grid);

      source.onchange = () => {
        const useImage = source.value === "image_file";
        promptLabel.classList.toggle("ai3d-hidden", useImage);
        imageFields.classList.toggle("ai3d-hidden", !useImage);
      };
      function updateModeFields() {
        const lowPoly = type.value === "LowPoly";
        const geometry = type.value === "Geometry";
        const lowPolyOption = type.querySelector('option[value="LowPoly"]');
        if (lowPolyOption) lowPolyOption.disabled = model.value === "3.1";
        if (model.value === "3.1" && lowPoly) type.value = "Normal";
        face.disabled = type.value === "LowPoly";
        pbr.disabled = geometry;
        if (geometry) pbr.checked = false;
        document.querySelectorAll("[data-ai3d-view-label]").forEach(label => {
          const requires31 = label.dataset.minModel === "3.1";
          const disabled = requires31 && model.value !== "3.1";
          label.classList.toggle("ai3d-hidden", disabled);
          const input = label.querySelector("input");
          if (input) input.disabled = disabled;
        });
        const help = document.getElementById("ai3dModeHelp");
        const modeHelp = {
          Normal: zh ? "带纹理的通用几何模型，适合大多数仿真资产。" : "Textured general geometry for most simulation assets.",
          LowPoly: zh ? "智能拓扑低模；面数参数不生效，且 3.1 不支持。" : "Retopologized low-poly model; face count is ignored and 3.1 is unsupported.",
          Geometry: zh ? "无纹理白模；PBR 参数不生效。" : "Untextured geometry; PBR is ignored.",
          Sketch: zh ? "面向草图或线稿输入，可结合提示词。" : "Designed for sketches or line art and can use a prompt.",
        };
        if (help) help.textContent = modeHelp[type.value] || "";
      }
      type.onchange = updateModeFields;
      model.onchange = updateModeFields;
      updateModeFields();

      const generate = el("button", {id: "ai3dGenerate", type: "button", text: zh ? "生成3D模型" : "Generate 3D Model"});
      generate.onclick = () => submitHunyuan3D(generate);
      panel.appendChild(el("div", {class: "ai3d-actions"}, [generate]));
      const recoverInput = el("input", {id: "ai3dRecoverId", type: "text"});
      recoverInput.placeholder = zh ? "已有腾讯任务号，可在此恢复查询和下载" : "Existing Tencent job ID to recover";
      const recoverButton = el("button", {type: "button", class: "secondary", text: zh ? "恢复已有任务" : "Recover Job"});
      recoverButton.onclick = () => recoverHunyuan3D(recoverButton);
      panel.appendChild(el("div", {class: "ai3d-recover"}, [recoverInput, recoverButton]));
      panel.appendChild(el("div", {id: "ai3dNotice", class: "ai3d-notice ai3d-hidden"}));
      panel.appendChild(el("div", {id: "ai3dJobList", class: "ai3d-job-list"}));
      panel.appendChild(el("div", {class: "ai3d-library"}, [
        el("h4", {text: zh ? "已生成的 AI3D 资产" : "Generated AI3D Assets"}),
        el("div", {id: "ai3dAssetList", class: "ai3d-asset-list"})
      ]));
      const physicsPanel = el("section", {id: "ai3dPhysicsPanel", class: "ai3d-physics ai3d-hidden"});
      physicsPanel.appendChild(el("div", {class: "ai3d-physics-heading"}, [
        el("div", {}, [
          el("h4", {text: zh ? "物理属性与 USD 生成" : "Physics Properties and USD Generation"}),
          el("p", {text: zh ? "为生成的 USDZ 设置质量、碰撞体、摩擦系数和刚体类型，并在同一文件夹生成同名 USD。" : "Configure mass, collision, friction, and body type, then create a same-name USD beside the USDZ."})
        ])
      ]));
      const physicsGrid = el("div", {class: "ai3d-physics-grid"});
      const physicsAssetLabel = el("label", {}, [
        el("span", {text: zh ? "选择 USDZ 资产" : "Select USDZ Asset"})
      ]);
      physicsAssetLabel.appendChild(el("select", {id: "ai3dPhysicsAsset"}));
      physicsGrid.appendChild(physicsAssetLabel);

      const massLabel = el("label", {}, [
        el("span", {text: zh ? "质量（kg）" : "Mass (kg)"})
      ]);
      massLabel.appendChild(el("input", {
        id: "ai3dPhysicsMass",
        type: "number",
        min: "0.0001",
        step: "0.01",
        value: "1"
      }));
      massLabel.appendChild(el("small", {
        class: "field-help",
        text: zh ? "动态刚体使用该质量；静态刚体会保留数值，但仿真中不受质量影响。" : "Used by dynamic bodies; static bodies retain the value but are unaffected by mass."
      }));
      physicsGrid.appendChild(massLabel);

      const collisionLabel = el("label", {}, [
        el("span", {text: zh ? "碰撞体类型" : "Collision Type"})
      ]);
      const collision = el("select", {id: "ai3dCollisionType"});
      [
        ["bounding_box", zh ? "包围盒（推荐，速度最快）" : "Bounding Box (recommended, fastest)"],
        ["convex_hull", zh ? "凸包" : "Convex Hull"],
        ["convex_decomposition", zh ? "凸分解（更贴合）" : "Convex Decomposition (closer fit)"],
        ["triangle_mesh", zh ? "三角网格（仅静态）" : "Triangle Mesh (static only)"],
      ].forEach(([value, text]) => collision.appendChild(el("option", {value, text})));
      collisionLabel.appendChild(collision);
      physicsGrid.appendChild(collisionLabel);

      const frictionLabel = el("label", {}, [
        el("span", {text: zh ? "摩擦系数" : "Friction Coefficient"})
      ]);
      frictionLabel.appendChild(el("input", {
        id: "ai3dPhysicsFriction",
        type: "number",
        min: "0",
        max: "10",
        step: "0.05",
        value: "0.5"
      }));
      frictionLabel.appendChild(el("small", {
        class: "field-help",
        text: zh ? "同时写入静摩擦和动摩擦系数。" : "Applied to both static and dynamic friction."
      }));
      physicsGrid.appendChild(frictionLabel);

      const bodyLabel = el("label", {}, [
        el("span", {text: zh ? "刚体类型" : "Rigid Body Type"})
      ]);
      const bodyType = el("select", {id: "ai3dBodyType"});
      bodyType.appendChild(el("option", {value: "static", text: zh ? "静态刚体" : "Static Body"}));
      bodyType.appendChild(el("option", {value: "dynamic", text: zh ? "动态刚体（受场景重力影响）" : "Dynamic Body (uses scene gravity)"}));
      bodyLabel.appendChild(bodyType);
      physicsGrid.appendChild(bodyLabel);
      physicsPanel.appendChild(physicsGrid);

      const physicsButton = el("button", {
        id: "ai3dCreatePhysics",
        type: "button",
        text: zh ? "生成 / 更新 USD" : "Create / Update USD"
      });
      physicsButton.onclick = () => createHunyuan3DPhysicsAsset(physicsButton);
      physicsPanel.appendChild(el("div", {class: "ai3d-actions"}, [physicsButton]));
      physicsPanel.appendChild(el("div", {id: "ai3dPhysicsNotice", class: "ai3d-notice ai3d-hidden"}));
      bodyType.onchange = () => {
        const triangleOption = collision.querySelector('option[value="triangle_mesh"]');
        const dynamic = bodyType.value === "dynamic";
        if (triangleOption) triangleOption.disabled = dynamic;
        if (dynamic && collision.value === "triangle_mesh") collision.value = "bounding_box";
      };
      panel.appendChild(physicsPanel);
      return panel;
    }
    function formatFileSize(bytes) {
      const value = Number(bytes || 0);
      if (value < 1024) return `${value} B`;
      if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
      return `${(value / (1024 * 1024)).toFixed(1)} MB`;
    }
    async function loadHunyuan3DAssets() {
      const list = document.getElementById("ai3dAssetList");
      if (!list) return;
      try {
        const data = await api("/api/hunyuan3d/assets", {});
        const output = document.getElementById("ai3dOutputDir");
        if (output) output.textContent = data.output_dir || "";
        list.innerHTML = "";
        const assets = asArray(data.assets);
        updateHunyuan3DPhysicsAssets(assets);
        if (!assets.length) {
          list.appendChild(el("div", {class: "asset-empty", text: state.lang === "zh" ? "还没有生成 USDZ 模型。" : "No generated USDZ models yet."}));
          return;
        }
        assets.forEach(asset => {
          const modified = asset.modified_at ? new Date(Number(asset.modified_at) * 1000).toLocaleString() : "";
          list.appendChild(el("div", {class: "ai3d-asset"}, [
            el("strong", {text: asset.name || ""}),
            el("div", {class: "ai3d-asset-paths"}, [
              el("code", {text: asset.path || ""}),
              ...(asset.usd_path ? [el("code", {text: asset.usd_path})] : [])
            ]),
            el("span", {
              class: "muted",
              text: `${asset.physics_ready ? (state.lang === "zh" ? "USD 已生成 · " : "USD ready · ") : ""}${formatFileSize(asset.size)}${modified ? ` · ${modified}` : ""}`
            })
          ]));
        });
      } catch (err) {
        list.textContent = `${state.lang === "zh" ? "读取资产失败" : "Failed to load assets"}: ${err.message}`;
      }
    }
    function updateHunyuan3DPhysicsAssets(assets) {
      const panel = document.getElementById("ai3dPhysicsPanel");
      const select = document.getElementById("ai3dPhysicsAsset");
      if (!panel || !select) return;
      const previous = select.value;
      select.innerHTML = "";
      asArray(assets).forEach(asset => {
        select.appendChild(el("option", {
          value: asset.path || "",
          text: `${asset.name || ""}${asset.physics_ready ? (state.lang === "zh" ? "（已有 USD）" : " (USD ready)") : ""}`
        }));
      });
      if (previous && asArray(assets).some(asset => asset.path === previous)) select.value = previous;
      panel.classList.toggle("ai3d-hidden", !assets.length);
    }
    async function createHunyuan3DPhysicsAsset(button) {
      const usdzPath = document.getElementById("ai3dPhysicsAsset")?.value || "";
      const mass = Number(document.getElementById("ai3dPhysicsMass")?.value || 0);
      const friction = Number(document.getElementById("ai3dPhysicsFriction")?.value || 0);
      const collisionType = document.getElementById("ai3dCollisionType")?.value || "bounding_box";
      const bodyType = document.getElementById("ai3dBodyType")?.value || "static";
      if (!usdzPath) {
        setHunyuan3DPhysicsNotice(state.lang === "zh" ? "请选择一个 USDZ 资产。" : "Select a USDZ asset.", true);
        return;
      }
      if (!(mass > 0)) {
        setHunyuan3DPhysicsNotice(state.lang === "zh" ? "质量必须大于 0 kg。" : "Mass must be greater than 0 kg.", true);
        return;
      }
      if (friction < 0 || friction > 10) {
        setHunyuan3DPhysicsNotice(state.lang === "zh" ? "摩擦系数必须在 0 到 10 之间。" : "Friction must be between 0 and 10.", true);
        return;
      }
      button.disabled = true;
      setHunyuan3DPhysicsNotice(
        state.lang === "zh" ? "正在创建带物理属性的 USD..." : "Creating the physics-enabled USD...",
        false,
        true
      );
      try {
        const data = await api("/api/hunyuan3d/physics", {
          usdz_path: usdzPath,
          mass,
          friction,
          collision_type: collisionType,
          body_type: bodyType,
        });
        setHunyuan3DPhysicsNotice(
          `${state.lang === "zh" ? "USD 已生成" : "USD created"}: ${data.result.usd_path}`,
          false
        );
        await loadHunyuan3DAssets();
      } catch (err) {
        setHunyuan3DPhysicsNotice(
          `${state.lang === "zh" ? "USD 生成失败" : "USD generation failed"}: ${err.message}`,
          true
        );
      } finally {
        button.disabled = false;
      }
    }
    function setHunyuan3DPhysicsNotice(message, isError=false, loading=false) {
      const notice = document.getElementById("ai3dPhysicsNotice");
      if (!notice) return;
      notice.className = `ai3d-notice${isError ? " error" : ""}${loading ? " loading" : ""}`;
      notice.textContent = message;
    }
    async function submitHunyuan3D(button) {
      const source = document.getElementById("ai3dSource")?.value || "text";
      const prompt = document.getElementById("ai3dPrompt")?.value.trim() || "";
      const useImage = source === "image_file";
      const imageFile = useImage ? (document.getElementById("ai3dImageFile")?.files?.[0] || null) : null;
      const assetName = document.getElementById("ai3dAssetName")?.value.trim() || "";
      if ((!useImage && !prompt) || (useImage && !imageFile)) {
        const message = source === "text"
          ? (state.lang === "zh" ? "请填写模型描述。" : "Enter a model prompt.")
          : (state.lang === "zh" ? "请选择主图 / 正视图。" : "Choose a primary / front image.");
        setHunyuan3DNotice(message, true);
        return;
      }
      const selectedViews = useImage
        ? Array.from(document.querySelectorAll("[data-ai3d-view]"))
          .filter(input => !input.disabled && input.files?.[0])
          .map(input => ({viewType: input.dataset.ai3dView, file: input.files[0]}))
        : [];
      const selectedFiles = [imageFile, ...selectedViews.map(item => item.file)].filter(Boolean);
      const invalidFile = selectedFiles.find(file => !["image/png", "image/jpeg"].includes(file.type));
      if (invalidFile) {
        setHunyuan3DNotice(state.lang === "zh" ? "图片只支持 PNG 或 JPG 格式。" : "Only PNG and JPG images are supported.", true);
        return;
      }
      const totalImageSize = selectedFiles.reduce((total, file) => total + file.size, 0);
      if (totalImageSize > 6 * 1024 * 1024) {
        setHunyuan3DNotice(state.lang === "zh" ? "所有图片原文件合计不能超过 6 MB。" : "The combined original image size must not exceed 6 MB.", true);
        return;
      }
      const faceValue = document.getElementById("ai3dFaceCount")?.value.trim() || "";
      button.disabled = true;
      button.textContent = state.lang === "zh" ? "正在提交..." : "Submitting...";
      setHunyuan3DNotice(
        state.lang === "zh" ? "正在读取输入并提交腾讯混元生3D任务，请稍等。" : "Preparing input and submitting the Hunyuan 3D job.",
        false,
        true
      );
      try {
        const imageBase64 = imageFile ? await readImageAsBase64(imageFile) : "";
        const multiViewImages = await Promise.all(selectedViews.map(async item => ({
          view_type: item.viewType,
          image_base64: await readImageAsBase64(item.file),
          image_name: item.file.name,
        })));
        const data = await api("/api/hunyuan3d/submit", {
          prompt: useImage ? "" : prompt,
          image_base64: useImage ? imageBase64 : "",
          image_name: useImage ? (imageFile?.name || "") : "",
          multi_view_images: useImage ? multiViewImages : [],
          asset_name: assetName,
          model: document.getElementById("ai3dModel")?.value || "3.0",
          generate_type: document.getElementById("ai3dGenerateType")?.value || "Normal",
          face_count: faceValue ? Number(faceValue) : "",
          enable_pbr: !!document.getElementById("ai3dPbr")?.checked,
        });
        setHunyuan3DNotice(
          state.lang === "zh" ? `任务已提交：${data.job.cloud_job_id}` : `Job submitted: ${data.job.cloud_job_id}`,
          false
        );
        renderHunyuan3DJob(data.job);
        startHunyuan3DPolling(data.job.id);
      } catch (err) {
        setHunyuan3DNotice(
          `${state.lang === "zh" ? "提交失败" : "Submission failed"}: ${err.message}`,
          true
        );
      } finally {
        button.disabled = false;
        button.textContent = state.lang === "zh" ? "生成3D模型" : "Generate 3D Model";
      }
    }
    function setHunyuan3DNotice(message, isError=false, loading=false) {
      const notice = document.getElementById("ai3dNotice");
      if (!notice) return;
      notice.className = `ai3d-notice${isError ? " error" : ""}${loading ? " loading" : ""}`;
      notice.textContent = message;
      notice.scrollIntoView({behavior: "smooth", block: "nearest"});
    }
    function readImageAsBase64(file) {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => {
          const value = String(reader.result || "");
          const comma = value.indexOf(",");
          if (comma < 0) {
            reject(new Error(state.lang === "zh" ? "无法读取图片内容。" : "Unable to read image data."));
            return;
          }
          resolve(value.slice(comma + 1));
        };
        reader.onerror = () => reject(reader.error || new Error(state.lang === "zh" ? "图片读取失败。" : "Image read failed."));
        reader.readAsDataURL(file);
      });
    }
    async function recoverHunyuan3D(button) {
      const cloudJobId = document.getElementById("ai3dRecoverId")?.value.trim() || "";
      if (!cloudJobId) {
        setHunyuan3DNotice(state.lang === "zh" ? "请填写已有腾讯任务号。" : "Enter an existing Tencent job ID.", true);
        return;
      }
      button.disabled = true;
      setHunyuan3DNotice(state.lang === "zh" ? "正在恢复并查询已有任务。" : "Recovering and querying the existing job.", false, true);
      try {
        const data = await api("/api/hunyuan3d/recover", {cloud_job_id: cloudJobId});
        renderHunyuan3DJob(data.job);
        startHunyuan3DPolling(data.job.id);
      } catch (err) {
        setHunyuan3DNotice(`${state.lang === "zh" ? "恢复失败" : "Recovery failed"}: ${err.message}`, true);
      } finally {
        button.disabled = false;
      }
    }
    function renderHunyuan3DJob(job) {
      const list = document.getElementById("ai3dJobList");
      if (!list) return;
      let card = list.querySelector(`[data-ai3d-job="${job.id}"]`);
      if (!card) {
        card = el("div", {class: "ai3d-job", "data-ai3d-job": job.id});
        list.prepend(card);
      }
      const statusMap = {
        submitted: state.lang === "zh" ? "已提交" : "Submitted",
        queued: state.lang === "zh" ? "排队中" : "Queued",
        running: state.lang === "zh" ? "生成中" : "Generating",
        completed: state.lang === "zh" ? "已完成" : "Completed",
        failed: state.lang === "zh" ? "生成失败" : "Failed",
        query_error: state.lang === "zh" ? "查询暂时失败，自动重试中" : "Status query failed, retrying",
        download_error: state.lang === "zh" ? "下载暂时失败，自动重试中" : "Download failed, retrying",
      };
      const progress = Math.max(0, Math.min(100, Number(job.progress || 0)));
      const completed = job.status === "completed";
      const exactProgress = job.progress_available === true || completed;
      const statusLabel = statusMap[job.status] || job.status;
      card.innerHTML = "";
      card.appendChild(el("div", {class: "progress-head"}, [
        el("strong", {text: exactProgress ? `${statusLabel} · ${progress.toFixed(0)}%` : statusLabel}),
        el("code", {text: job.cloud_job_id || ""})
      ]));
      card.appendChild(el("div", {class: "progress-track"}, [
        el("div", {
          class: `progress-fill${exactProgress ? "" : " indeterminate"}`,
          style: exactProgress ? `width:${progress}%` : ""
        })
      ]));
      if (!exactProgress && ["submitted", "queued", "running"].includes(job.status)) {
        card.appendChild(el("div", {
          class: "muted",
          text: state.lang === "zh"
            ? "腾讯查询接口只返回排队/生成中/完成状态，不提供实时百分比。"
            : "Tencent reports queued/running/completed states without an exact percentage."
        }));
      }
      if (job.prompt) card.appendChild(el("div", {class: "muted", text: job.prompt}));
      if (job.image_name) {
        card.appendChild(el("div", {
          class: "muted",
          text: `${state.lang === "zh" ? "主图" : "Primary image"}: ${job.image_name}`
        }));
      }
      const viewNameMap = {
        left: state.lang === "zh" ? "左视图" : "Left",
        right: state.lang === "zh" ? "右视图" : "Right",
        back: state.lang === "zh" ? "后视图" : "Back",
        top: state.lang === "zh" ? "顶视图" : "Top",
        bottom: state.lang === "zh" ? "底视图" : "Bottom",
        left_front: state.lang === "zh" ? "左前 45°" : "Left Front 45°",
        right_front: state.lang === "zh" ? "右前 45°" : "Right Front 45°",
      };
      asArray(job.multi_view_names).forEach(item => {
        card.appendChild(el("div", {
          class: "muted",
          text: `${viewNameMap[item.view_type] || item.view_type}: ${item.image_name || ""}`
        }));
      });
      if (job.asset_name) {
        card.appendChild(el("div", {
          class: "muted",
          text: `${state.lang === "zh" ? "资产名称" : "Asset name"}: ${job.asset_name}`
        }));
      }
      if (job.credit_consumed !== undefined && job.credit_consumed !== null && job.credit_consumed !== "") {
        card.appendChild(el("div", {
          class: "muted",
          text: `${state.lang === "zh" ? "消耗积分" : "Credits consumed"}: ${job.credit_consumed}`
        }));
      }
      if (job.error) card.appendChild(el("div", {class: "ai3d-error", text: job.error}));
      if (job.local_path) {
        card.appendChild(el("div", {class: "ai3d-result"}, [
          el("span", {text: state.lang === "zh" ? "本地模型：" : "Local model: "}),
          el("code", {text: job.local_path})
        ]));
      }
      if (job.preview_url) {
        card.appendChild(el("img", {class: "ai3d-preview", src: job.preview_url, alt: state.lang === "zh" ? "模型预览" : "Model preview"}));
      }
    }
    async function pollHunyuan3D(generationId) {
      try {
        const data = await api("/api/hunyuan3d/status", {generation_id: generationId});
        renderHunyuan3DJob(data.job);
        if (["completed", "failed"].includes(data.job.status)) {
          clearInterval(state.ai3dTimers[generationId]);
          delete state.ai3dTimers[generationId];
          if (data.job.status === "completed") loadHunyuan3DAssets();
        }
      } catch (err) {
        const card = document.querySelector(`[data-ai3d-job="${generationId}"]`);
        if (card) card.appendChild(el("div", {class: "ai3d-error", text: err.message}));
      }
    }
    function startHunyuan3DPolling(generationId) {
      if (state.ai3dTimers[generationId]) clearInterval(state.ai3dTimers[generationId]);
      pollHunyuan3D(generationId);
      state.ai3dTimers[generationId] = setInterval(() => pollHunyuan3D(generationId), 5000);
    }
    async function loadForm(action, params) {
      try {
        const data = await api("/api/form", {action, params});
        renderForm(data.form, action);
      } catch (err) { renderMessage(`<strong>${currentCopy().errorPrefix}:</strong>${escapeHtml(translatePlainText(err.message))}`); }
    }
    async function sendText() {
      if (state.runningJobs.length > 0) {
        await stopCurrentTask();
        return;
      }
      const text = document.getElementById("chatInput").value.trim();
      if (!text) return;
      document.getElementById("chatInput").value = "";
      setSteps(0, []);
      renderMessage(`<strong>${currentCopy().userPrefix}:</strong>${escapeHtml(text)}`);
      try {
        const data = await api("/api/intent", {text});
        if (data.action === "stop_active_job") {
          const stopped = await api("/api/stop", {});
          state.runningJobs = [];
          updateSendButton();
          renderMessage(`<strong>${currentCopy().agentPrefix}:</strong>${escapeHtml(translatePlainText(stopped.message))}`, {clearable: true, action: state.current});
          setSteps(3, [0,1,2,3]);
          return;
        }
        const formCopy = localizedAction(data.action, data.form);
        const recognized = state.lang === "zh"
          ? `${currentCopy().recognizedPrefix} ${escapeHtml(formCopy.title)}。${currentCopy().recognizedSuffix}`
          : `${currentCopy().recognizedPrefix} ${escapeHtml(formCopy.title)}. ${currentCopy().recognizedSuffix}`;
        renderMessage(`<strong>${currentCopy().agentPrefix}:</strong>${recognized}`);
        renderForm(data.form, data.action);
      } catch (err) { renderMessage(`<strong>${currentCopy().errorPrefix}:</strong>${escapeHtml(translatePlainText(err.message))}`); }
    }
    async function submitAction(action, form) {
      let loading = null;
      try {
        setSteps(2, [0,1]);
        const params = collectParams(form);
        setFormBusy(form, true);
        loading = showLoading(action);
        const data = await api("/api/launch", {action, params});
        state.jobs = data.jobs || [];
        state.runningJobs = state.jobs.map(j => j.id);
        updateSendButton();
        if (loading) loading.remove();
        renderJobs(data, action);
        setSteps(3, [0,1,2,3]);
      } catch (err) {
        if (loading) loading.remove();
        renderMessage(`<strong>${currentCopy().launchFailurePrefix}:</strong>${escapeHtml(translatePlainText(err.message))}`, {clearable: true, action});
      } finally {
        setFormBusy(form, false);
        updateConditionalFields(form);
      }
    }
    async function stopCurrentTask() {
      try {
        const stopped = await api("/api/stop", {});
        clearProgressTimers();
        state.runningJobs = [];
        updateSendButton();
        renderMessage(`<strong>${currentCopy().agentPrefix}:</strong>${escapeHtml(translatePlainText(stopped.message))}`, {clearable: true, action: state.current});
      } catch (err) {
        const label = state.lang === "zh" ? "停止失败" : "Stop failed";
        renderMessage(`<strong>${label}:</strong>${escapeHtml(translatePlainText(err.message))}`, {clearable: true, action: state.current});
      }
    }
    function statusText(status) {
      const map = {
        zh: {running: "运行中", completed: "已完成", error: "异常", stopped: "已停止", waiting: "等待中"},
        en: {running: "Running", completed: "Completed", error: "Error", stopped: "Stopped", waiting: "Waiting"},
      };
      const copy = map[state.lang] || map.en;
      return copy[status] || copy.waiting;
    }
    function pctText(value, digits=1) {
      const num = Number(value);
      if (!Number.isFinite(num)) return "0.0%";
      return `${num.toFixed(digits)}%`;
    }
    function renderProgressBox(job) {
      const box = el("div", {class: "inference-progress", "data-progress-id": job.id}, [
        el("div", {class: "progress-head"}, [
          el("span", {text: state.lang === "zh" ? "推理进度" : "Inference Progress"}),
          el("span", {class: "progress-percent", "data-progress-percent": job.id, text: state.lang === "zh" ? "等待启动" : "Waiting to start"})
        ]),
        el("div", {class: "progress-track"}, [
          el("div", {class: "progress-fill", "data-progress-fill": job.id})
        ]),
        el("div", {class: "progress-stats"}, [
          progressStat(job.id, "completed", state.lang === "zh" ? "已完成" : "Completed", "0/0"),
          progressStat(job.id, "success", state.lang === "zh" ? "成功" : "Success", "0", "success"),
          progressStat(job.id, "failure", state.lang === "zh" ? "失败" : "Failure", "0", "failure"),
          progressStat(job.id, "rate", state.lang === "zh" ? "成功率" : "Success Rate", "0.0%")
        ]),
        el("div", {class: "progress-foot", "data-progress-foot": job.id, text: state.lang === "zh" ? "推理客户端正在启动，等待第一条进度。" : "The inference client is starting. Waiting for the first progress update."})
      ]);
      return box;
    }
    function progressStat(jobId, key, label, value, valueClass="") {
      return el("div", {class: "progress-stat"}, [
        el("div", {class: "progress-label", text: label}),
        el("div", {class: `progress-value ${valueClass}`.trim(), "data-progress-stat": `${jobId}:${key}`, text: value})
      ]);
    }
    function setProgressText(jobId, key, value) {
      const node = document.querySelector(`[data-progress-stat="${jobId}:${key}"]`);
      if (node) node.textContent = value;
    }
    function updateInferenceProgress(jobId, progress) {
      const total = Number(progress.total_episodes || 0);
      const completed = Number(progress.completed || 0);
      const success = Number(progress.success || 0);
      const failure = Number(progress.failure || 0);
      const percent = Math.max(0, Math.min(100, Number(progress.percent || 0)));
      const rate = Math.max(0, Math.min(1, Number(progress.success_rate || 0)));
      const fill = document.querySelector(`[data-progress-fill="${jobId}"]`);
      const percentNode = document.querySelector(`[data-progress-percent="${jobId}"]`);
      const foot = document.querySelector(`[data-progress-foot="${jobId}"]`);
      if (fill) fill.style.width = `${percent}%`;
      if (percentNode) percentNode.textContent = `${statusText(progress.status)} · ${pctText(percent)}`;
      setProgressText(jobId, "completed", `${completed}/${total}`);
      setProgressText(jobId, "success", String(success));
      setProgressText(jobId, "failure", String(failure));
      setProgressText(jobId, "rate", pctText(rate * 100));
      if (foot) {
        let episode = state.lang === "zh"
          ? (progress.current_episode ? `当前回合 ${progress.current_episode}` : "等待回合开始")
          : (progress.current_episode ? `Current episode ${progress.current_episode}` : "Waiting for episode start");
        if (progress.status === "completed") episode = state.lang === "zh" ? "全部回合完成" : "All episodes completed";
        if (progress.status === "error") episode = state.lang === "zh" ? "推理异常结束" : "Inference ended with an error";
        if (progress.status === "stopped") episode = state.lang === "zh" ? "推理已停止" : "Inference stopped";
        const last = progress.last_outcome
          ? (state.lang === "zh" ? `；上一回合：${progress.last_outcome} (${progress.last_reason || "无原因"})` : `; last episode: ${progress.last_outcome} (${progress.last_reason || "no reason"})`)
          : "";
        foot.textContent = `${episode}${last}`;
      }
    }
    async function pollProgress(jobId) {
      try {
        const data = await api("/api/progress", {job_id: jobId});
        const foot = document.querySelector(`[data-progress-foot="${jobId}"]`);
        if (!data.ready) {
          if (foot) foot.textContent = data.message || "等待进度数据。";
          return;
        }
        updateInferenceProgress(jobId, data.progress);
        if (["completed", "error", "stopped"].includes(data.progress.status)) {
          clearInterval(state.progressTimers[jobId]);
          delete state.progressTimers[jobId];
        }
      } catch (err) {
        const foot = document.querySelector(`[data-progress-foot="${jobId}"]`);
        if (foot) foot.textContent = `进度读取失败：${err.message}`;
      }
    }
    function startProgressPolling(jobId) {
      if (state.progressTimers[jobId]) clearInterval(state.progressTimers[jobId]);
      pollProgress(jobId);
      state.progressTimers[jobId] = setInterval(() => pollProgress(jobId), 1500);
    }
    function renderInlineStatsBox(job, action, scope, items) {
      return el("div", {class: "progress-stats augment-stats run-output", "data-output-action": action, [`data-${scope}-id`]: job.id},
        items.map(item => inlineStat(job.id, scope, item.key, item.label))
      );
    }
    function inlineStat(jobId, scope, key, label) {
      return el("div", {class: "progress-stat"}, [
        el("div", {class: "progress-label", text: label}),
        el("div", {class: "progress-value", [`data-${scope}-stat`]: `${jobId}:${key}`, text: "0"})
      ]);
    }
    function setInlineStat(jobId, scope, key, value) {
      const node = document.querySelector(`[data-${scope}-stat="${jobId}:${key}"]`);
      if (node) node.textContent = String(value ?? 0);
    }
    function mountInlineStats(job, action, scope, items) {
      const formPanel = document.querySelector(".form-panel");
      if (!formPanel) return;
      formPanel.querySelectorAll(`.augment-stats.run-output[data-output-action="${action}"]`).forEach(node => node.remove());
      const box = renderInlineStatsBox(job, action, scope, items);
      const actions = formPanel.querySelector(".actions");
      if (actions) formPanel.insertBefore(box, actions);
      else formPanel.appendChild(box);
    }
    async function pollAugmentStats(jobId) {
      try {
        const data = await api("/api/augment-stats", {job_id: jobId});
        const stats = data.stats || {};
        setInlineStat(jobId, "augment", "success", stats.success);
        setInlineStat(jobId, "augment", "failure", stats.failure);
        setInlineStat(jobId, "augment", "total", stats.total);
        if (stats.done) {
          clearInterval(state.progressTimers[`augment:${jobId}`]);
          delete state.progressTimers[`augment:${jobId}`];
        }
      } catch (err) {
        clearInterval(state.progressTimers[`augment:${jobId}`]);
        delete state.progressTimers[`augment:${jobId}`];
      }
    }
    function startAugmentStatsPolling(jobId) {
      const key = `augment:${jobId}`;
      if (state.progressTimers[key]) clearInterval(state.progressTimers[key]);
      pollAugmentStats(jobId);
      state.progressTimers[key] = setInterval(() => pollAugmentStats(jobId), 1500);
    }
    async function pollInferenceStats(jobId) {
      try {
        const data = await api("/api/inference-stats", {job_id: jobId});
        const stats = data.stats || {};
        setInlineStat(jobId, "inference", "success", stats.success);
        setInlineStat(jobId, "inference", "failure", stats.failure);
        setInlineStat(jobId, "inference", "total", stats.total);
        if (stats.done) {
          clearInterval(state.progressTimers[`inference:${jobId}`]);
          delete state.progressTimers[`inference:${jobId}`];
        }
      } catch (err) {
        clearInterval(state.progressTimers[`inference:${jobId}`]);
        delete state.progressTimers[`inference:${jobId}`];
      }
    }
    function startInferenceStatsPolling(jobId) {
      const key = `inference:${jobId}`;
      if (state.progressTimers[key]) clearInterval(state.progressTimers[key]);
      pollInferenceStats(jobId);
      state.progressTimers[key] = setInterval(() => pollInferenceStats(jobId), 1500);
    }
    async function pollConvertProgress(jobId) {
      try {
        const data = await api("/api/convert-progress", {job_id: jobId});
        const progress = data.progress || {};
        setInlineStat(jobId, "convert", "completed", progress.completed);
        setInlineStat(jobId, "convert", "total", progress.total);
        setInlineStat(jobId, "convert", "kept_frames", progress.kept_frames);
        if (progress.done) {
          clearInterval(state.progressTimers[`convert:${jobId}`]);
          delete state.progressTimers[`convert:${jobId}`];
        }
      } catch (err) {
        clearInterval(state.progressTimers[`convert:${jobId}`]);
        delete state.progressTimers[`convert:${jobId}`];
      }
    }
    function startConvertProgressPolling(jobId) {
      const key = `convert:${jobId}`;
      if (state.progressTimers[key]) clearInterval(state.progressTimers[key]);
      pollConvertProgress(jobId);
      state.progressTimers[key] = setInterval(() => pollConvertProgress(jobId), 1500);
    }
    function asArray(value) {
      return Array.isArray(value) ? value : [];
    }
    function renderAssetTags(files) {
      const tags = el("div", {class: "asset-tags"});
      const values = asArray(files).filter(Boolean);
      if (!values.length) {
        tags.appendChild(el("span", {class: "muted", text: "未匹配到已有文件"}));
        return tags;
      }
      values.forEach(file => tags.appendChild(el("span", {class: "asset-tag", text: String(file)})));
      return tags;
    }
    function renderAssetSection(title, rows, columns) {
      const section = el("div", {class: "asset-section"});
      section.appendChild(el("h3", {text: title}));
      if (!rows.length) {
        section.appendChild(el("div", {class: "asset-empty", text: "暂无"}));
        return section;
      }
      rows.forEach(row => {
        const line = el("div", {class: "asset-row"});
        columns.forEach(col => {
          const value = row[col.key];
          if (col.kind === "tags") {
            line.appendChild(renderAssetTags(value));
          } else {
            line.appendChild(el(col.strong ? "strong" : "div", {class: col.strong ? "" : "muted", text: String(value || "")}));
          }
        });
        section.appendChild(line);
      });
      return section;
    }
    function renderAssetFileList(title, files) {
      const section = el("div", {class: "asset-section"});
      section.appendChild(el("h3", {text: title}));
      const content = el("div", {class: "asset-empty"});
      content.appendChild(renderAssetTags(files));
      section.appendChild(content);
      return section;
    }
    function renderAssetPlan(plan) {
      const box = el("div", {class: "asset-plan"});
      box.appendChild(el("div", {class: "asset-summary", text: plan.summary || "已完成资产分析。"}));
      if (plan.api_warning) box.appendChild(el("div", {class: "asset-summary", text: plan.api_warning}));
      box.appendChild(el("div", {class: "asset-summary", html: `资产目录：<code>${escapeHtml(plan.asset_dir || "")}</code>；扫描到 <strong>${asArray(plan.all_assets).length}</strong> 个 USD 资产。`}));
      box.appendChild(renderAssetFileList("当前拥有的全部 USD 资产", asArray(plan.all_assets)));
      box.appendChild(renderAssetSection("需要的资产与匹配结果", asArray(plan.needed_assets), [
        {key: "name", strong: true},
        {key: "purpose"},
        {key: "matched_files", kind: "tags"},
      ]));
      const ownedRows = asArray(plan.owned_assets).map(row => ({
        ...row,
        matched_files: row.matched_files || (row.file ? [row.file] : []),
      }));
      box.appendChild(renderAssetSection("可直接使用的已有资产", ownedRows, [
        {key: "file", strong: true},
        {key: "use"},
        {key: "matched_files", kind: "tags"},
      ]));
      box.appendChild(renderAssetSection("仍缺失或建议补充的资产", asArray(plan.missing_assets), [
        {key: "name", strong: true},
        {key: "reason"},
        {key: "matched_files", kind: "tags"},
      ]));
      const suggestions = asArray(plan.suggestions).filter(Boolean);
      if (suggestions.length) {
        box.appendChild(renderAssetSection("建议", suggestions.map(item => ({name: item, reason: ""})), [
          {key: "name", strong: true},
          {key: "reason"},
          {key: "matched_files", kind: "tags"},
        ]));
      }
      return box;
    }
    function renderJobs(data, action) {
      clearProgressTimers();
      document.querySelectorAll(".job-panel.run-output").forEach(panel => {
        if (!action || panel.getAttribute("data-output-action") === action) panel.remove();
      });
      if (action === "augment_data") {
        (data.jobs || []).forEach(j => {
          if (j.log_path) {
            mountInlineStats(j, action, "augment", [
              {key: "success", label: state.lang === "zh" ? "成功样本数量" : "Successful Samples"},
              {key: "failure", label: state.lang === "zh" ? "失败样本数量" : "Failed Samples"},
              {key: "total", label: state.lang === "zh" ? "总数量" : "Total"}
            ]);
            startAugmentStatsPolling(j.id);
          }
        });
        return;
      }
      if (action === "run_inference") {
        (data.jobs || []).forEach(j => {
          if (j.kind === "inference" && j.log_path) {
            mountInlineStats(j, action, "inference", [
              {key: "success", label: state.lang === "zh" ? "成功数量" : "Successful Episodes"},
              {key: "failure", label: state.lang === "zh" ? "失败数量" : "Failed Episodes"},
              {key: "total", label: state.lang === "zh" ? "总数量" : "Total"}
            ]);
            startInferenceStatsPolling(j.id);
          }
        });
        return;
      }
      if (action === "convert_lerobot") {
        (data.jobs || []).forEach(j => {
          if (j.log_path) {
            mountInlineStats(j, action, "convert", [
              {key: "completed", label: state.lang === "zh" ? "已处理 episode" : "Processed Episodes"},
              {key: "total", label: state.lang === "zh" ? "总 episode" : "Total Episodes"},
              {key: "kept_frames", label: state.lang === "zh" ? "保留帧数" : "Kept Frames"}
            ]);
            startConvertProgressPolling(j.id);
          }
        });
        return;
      }
      const panel = el("div", {class: "job-panel run-output", "data-output-action": action || ""});
      panel.appendChild(el("h2", {text: translatePlainText(data.message || currentCopy().completed)}));
      if (data.registered) {
        panel.appendChild(el("p", {html: `${currentCopy().taskId}: <code>${data.registered.task_id}</code>, ${currentCopy().dataset}: <code>${data.registered.dataset_file}</code>`}));
      }
      if (data.asset_plan) {
        panel.appendChild(renderAssetPlan(data.asset_plan));
      }
      (data.jobs || []).forEach(j => {
        const row = el("div", {class: "job"});
        row.appendChild(el("div", {html: `<strong>${escapeHtml(translatePlainText(j.title))}</strong> <code>${j.id}</code>`}));
        row.appendChild(el("code", {text: j.command}));
        if (j.kind === "inference" && j.progress_path) {
          row.appendChild(renderProgressBox(j));
          startProgressPolling(j.id);
        }
        panel.appendChild(row);
      });
      document.getElementById("workspace").prepend(panel);
    }
    async function boot() {
      const res = await fetch("/api/bootstrap");
      const data = await res.json();
      state.actions = data.actions; state.tasks = data.tasks;
      renderNav(); setSteps(0, []); updateSendButton();
    }
    document.getElementById("langToggle").onclick = () => setLanguage(state.lang === "en" ? "zh" : "en");
    document.getElementById("sendBtn").onclick = sendText;
    document.getElementById("chatInput").addEventListener("keydown", e => { if (e.key === "Enter") sendText(); });
    setLanguage("en");
    boot();
