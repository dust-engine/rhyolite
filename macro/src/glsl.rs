use syn::Lit;

#[cfg(feature = "glsl")]
pub fn glsl(input: proc_macro2::TokenStream) -> proc_macro2::TokenStream {
    let input = match syn::parse2::<syn::LitStr>(input) {
        Ok(input) => input,
        Err(err) => return err.to_compile_error(),
    };
    let mut path = proc_macro::Span::call_site().source_file().path();
    path.pop();
    path.push(input.value());
    let path = path.as_path();

    proc_macro::tracked_path::path(&path.as_os_str().to_str().unwrap());

    let shader_stage = path.extension()
        .and_then(|extension| extension.to_str())
        .and_then(|extension| Some(match extension {
            "vert" => shaderc::ShaderKind::Vertex,
            "frag" => shaderc::ShaderKind::Fragment,
            "comp" => shaderc::ShaderKind::Compute,
            "geom" => shaderc::ShaderKind::Geometry,
            "tesc" => shaderc::ShaderKind::TessControl,
            "tese" => shaderc::ShaderKind::TessEvaluation,
            "mesh" => shaderc::ShaderKind::Mesh,
            "task" => shaderc::ShaderKind::Task,
            "rint" => shaderc::ShaderKind::Intersection,
            "rgen" => shaderc::ShaderKind::RayGeneration,
            "rmiss" => shaderc::ShaderKind::Miss,
            "rcall" => shaderc::ShaderKind::Callable,
            "rahit" => shaderc::ShaderKind::AnyHit,
            "rchit" => shaderc::ShaderKind::ClosestHit,
            _ => shaderc::ShaderKind::InferFromSource,
        }))
        .unwrap_or(shaderc::ShaderKind::InferFromSource);

    let file = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(err) => {
            let err = err.to_string();
            return quote::quote_spanned! { input.span()=>
                compile_error!("GLSL Shader file not found")
            };
        }
    };
    let source = match std::io::read_to_string(file) {
        Ok(source) => source,
        Err(err) => {
            let err = err.to_string();
            return quote::quote_spanned! { input.span()=>
                compile_error!("Cannot open GLSL shader file")
            };
        }
    };

    let compiler = shaderc::Compiler::new().unwrap();
    let mut options = shaderc::CompileOptions::new().unwrap();
    options.set_target_spirv(shaderc::SpirvVersion::V1_3);
    options.set_target_env(shaderc::TargetEnv::Vulkan, (1 << 22) | (3 << 12));


    let binary_result = compiler.compile_into_spirv(
        &source, shader_stage,
        path.file_name().and_then(|f| f.to_str()).unwrap_or("unknown.glsl"),
        "main",
        Some(&options)
    );

    let binary_result = match binary_result {
        Ok(binary) => {
            binary
        },
        Err(err) => {
            let err = err.to_string();
            input.span().unwrap().error(err).emit();
            return quote::quote! {
                ::rhyolite::shader::SpirvShader {
                    data: &[]
                }
            };
        }
    };

    let binary = binary_result.as_binary();
    let reflect_result = spirq::ReflectConfig::new()
    .spv(binary)
    .combine_img_samplers(true)
    .reflect().unwrap();
    let reflect_debug = reflect_result.iter().map(|result| {
        match result.vars[0].name()
    })

    let bin = U32Slice(binary);
    return quote::quote! {
        ::rhyolite::shader::SpirvShader {
            data: #bin
        }
    }
}

struct U32Slice<'a>(&'a [u32]);
impl<'a> quote::ToTokens for U32Slice<'a> {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        use proc_macro2::{TokenTree, Punct, Spacing, Group, Delimiter, TokenStream, Literal};
        tokens.extend_one(TokenTree::Punct(Punct::new('&', Spacing::Alone)));
        tokens.extend_one(TokenTree::Group(Group::new(Delimiter::Bracket, {
            TokenStream::from_iter(self.0.iter().flat_map(|num| {
                std::iter::once(TokenTree::Literal(Literal::u32_unsuffixed(*num)))
                .chain(std::iter::once(TokenTree::Punct(Punct::new(',', Spacing::Alone))))
            }))
        })));
    }
}
