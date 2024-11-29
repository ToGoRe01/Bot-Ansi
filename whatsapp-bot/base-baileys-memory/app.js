const { createBot, createProvider, createFlow, addKeyword } = require('@bot-whatsapp/bot')
const axios = require('axios');
const { url, usuario, contra, loginUri } = require('./environment')

const QRPortalWeb = require('@bot-whatsapp/portal')
const BaileysProvider = require('@bot-whatsapp/provider/baileys')
const MockAdapter = require('@bot-whatsapp/database/mock')

const userResponses = [];  // Array para almacenar las respuestas del usuario
let salir = true;
let token = '';
let config = {};

const flowTest = addKeyword(['test', 'cuestionario', 'ansiedad'])
    .addAction(async (_, { flowDynamic }) => {
        await flowDynamic('🎉 ¡Hola! Este es un test de ansiedad. Responde con "sí" o "no". ¿Quieres comenzar?');
    })
    .addAction(
        { capture: true }, // Captura la respuesta del usuario
        async (ctx, { flowDynamic, endFlow }) => {
            const input = ctx.body?.trim().toLowerCase(); // Normaliza la respuesta del usuario
            const registrado = (await axios.post(`${url}/api/login/find`, {
                numero_celular: ctx.from.slice(2)
            }, config)).data.data;
            if (['sí', 'si'].includes(input)) {
                if (registrado == null) {
                    salir = true;
                    await flowDynamic(`Tu número no se encuentra registrado. Por favor, registra tu número en nuestro sistema para poder realizar este test, mediante el siguiente link ${loginUri}`);
                } else {
                    salir = false;// Si responde sí, continúa con la primera pregunta
                    await flowDynamic('📝 Perfecto. Empecemos con la primera pregunta: ¿Has sentido temblor en las piernas?');
                }

            } else if (['no'].includes(input)) {
                salir = true;// Si responde no, termina el flujo
                await flowDynamic('👋 Entendido. Si necesitas ayuda, no dudes en escribirnos.');
                // Termina el flujo si la respuesta es "no".
            } else {
                // Si la respuesta es algo diferente a sí/no, vuelve a preguntar
                await flowDynamic('⚠️ Por favor, responde solo con "sí" o "no". ¿Quieres comenzar el test?');
            }
        }
    )
    .addAction(
        { capture: true }, // Captura la respuesta para la primera pregunta
        async (ctx, { flowDynamic, endFlow }) => {
            if (salir) {
                return endFlow();
            } else {
                const input = ctx.body?.trim(); // Captura la respuesta abierta
                userResponses.push({ pregunta: '¿Has sentido temblor en las piernas?', respuesta: input });
                await flowDynamic('😌 ¿Te resulta difícil relajarte?');  // Pregunta siguiente
            }

        }
    )
    .addAction(
        { capture: true }, // Captura la respuesta para la segunda pregunta
        async (ctx, { flowDynamic }) => {
            const input = ctx.body?.trim(); // Captura la respuesta abierta
            userResponses.push({ pregunta: '¿Te resulta difícil relajarte?', respuesta: input });
            await flowDynamic('💭 ¿Tienes temor a que ocurra lo peor?');  // Pregunta siguiente
        }
    )
    .addAction(
        { capture: true }, // Captura la respuesta para la tercera pregunta
        async (ctx, { flowDynamic }) => {
            const input = ctx.body?.trim(); // Captura la respuesta abierta
            userResponses.push({ pregunta: '¿Tienes temor a que ocurra lo peor?', respuesta: input });
            await flowDynamic('💓 ¿Tienes latidos del corazón fuertes y acelerados?');  // Pregunta siguiente
        }
    )
    .addAction(
        { capture: true }, // Captura la respuesta para la cuarta pregunta
        async (ctx, { flowDynamic }) => {
            const input = ctx.body?.trim(); // Captura la respuesta abierta
            userResponses.push({ pregunta: '¿Tienes latidos del corazón fuertes y acelerados?', respuesta: input });
            await flowDynamic('🌀 ¿Te sientes inestable?');  // Pregunta siguiente
        }
    )
    .addAction(
        { capture: true }, // Captura la respuesta para la quinta pregunta
        async (ctx, { flowDynamic }) => {
            const input = ctx.body?.trim(); // Captura la respuesta abierta
            userResponses.push({ pregunta: '¿Te sientes inestable?', respuesta: input });
            await flowDynamic('😨 ¿Te sientes atemorizado o asustado?');  // Pregunta siguiente
        }
    )
    .addAction(
        { capture: true }, // Captura la respuesta para la sexta pregunta
        async (ctx, { flowDynamic }) => {
            const input = ctx.body?.trim(); // Captura la respuesta abierta
            userResponses.push({ pregunta: '¿Te sientes atemorizado o asustado?', respuesta: input });
            await flowDynamic('😰 ¿Te sientes nervioso?');  // Pregunta siguiente
        }
    )
    .addAction(
        { capture: true }, // Captura la respuesta para la séptima pregunta
        async (ctx, { flowDynamic }) => {
            const input = ctx.body?.trim(); // Captura la respuesta abierta
            userResponses.push({ pregunta: '¿Te sientes nervioso?', respuesta: input });
            await flowDynamic('🤲 ¿Has sentido temblores en las manos?');  // Pregunta siguiente
        }
    )
    .addAction(
        { capture: true }, // Captura la respuesta para la octava pregunta
        async (ctx, { flowDynamic }) => {
            const input = ctx.body?.trim(); // Captura la respuesta abierta
            userResponses.push({ pregunta: '¿Has sentido temblores en las manos?', respuesta: input });
            await flowDynamic('😟 ¿Te sientes inquieto o inseguro?');  // Pregunta siguiente
        }
    )
    .addAction(
        { capture: true }, // Captura la respuesta para la novena pregunta
        async (ctx, { flowDynamic }) => {
            const input = ctx.body?.trim(); // Captura la respuesta abierta
            userResponses.push({ pregunta: '¿Te sientes inquieto o inseguro?', respuesta: input });
            await flowDynamic('😱 ¿Tienes miedo a perder el control?');  // Pregunta siguiente
        }
    )
    .addAction(
        { capture: true }, // Captura la respuesta para la décima pregunta
        async (ctx, { flowDynamic }) => {
            const input = ctx.body?.trim(); // Captura la respuesta abierta
            userResponses.push({ pregunta: '¿Tienes miedo a perder el control?', respuesta: input });
            await flowDynamic('😧 ¿Tienes sensación de ahogo?');  // Pregunta siguiente
        }
    )
    .addAction(
        { capture: true }, // Captura la respuesta para la undécima pregunta
        async (ctx, { flowDynamic }) => {
            const input = ctx.body?.trim(); // Captura la respuesta abierta
            userResponses.push({ pregunta: '¿Tienes sensación de ahogo?', respuesta: input });
            await flowDynamic('💔 ¿Sientes miedo en general?');  // Pregunta siguiente
        }
    )
    .addAction(
        { capture: true }, // Captura la respuesta para la duodécima pregunta
        async (ctx, { flowDynamic }) => {
            const input = ctx.body?.trim(); // Captura la respuesta abierta
            userResponses.push({ pregunta: '¿Sientes miedo en general?', respuesta: input });
            await flowDynamic('💪 ¿Tienes problemas digestivos?');  // Pregunta siguiente
        }
    )
    .addAction(
        { capture: true }, // Captura la respuesta para la última pregunta
        async (ctx, { flowDynamic }) => {
            const input = ctx.body?.trim(); // Captura la respuesta abierta
            userResponses.push({ pregunta: '¿Tienes problemas digestivos?', respuesta: input });
            await flowDynamic('✅ Gracias por completar el cuestionario. Si necesitas más ayuda, no dudes en escribirnos.');
            console.log(userResponses);
        }
    );


module.exports = flowTest;

const flowPrincipal = addKeyword('Hola', 'Comenzar', 'kkkk') // El asterisco captura cualquier palabra o mensaje
    .addAnswer('🙌 Hola bienvenido a este *Chatbot* de autoayuda')
    .addAnswer('Para comenzar por favor utiliza alguno de los siguientes comandos: \'test\', \'cuestionario\' o \'ansiedad\'',
        null,
        null,
        [flowTest] // Esto sigue activando el siguiente flujo
    );

const main = async () => {
    const adapterDB = new MockAdapter()
    const adapterFlow = createFlow([flowPrincipal])
    const adapterProvider = createProvider(BaileysProvider)

    createBot({
        flow: adapterFlow,
        provider: adapterProvider,
        database: adapterDB,
    })

    QRPortalWeb()
}

const getToken = async () => {
    try {
        token = (await axios.post(`${url}/api/auth/get-token`, {
            username: usuario,
            password: contra
        })).data.token;
        config = {
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${token}`
            }
        };
    } catch (error) {
        console.log(`Error en la petición: ${error.message}`)
    }
    main();
}

getToken()
