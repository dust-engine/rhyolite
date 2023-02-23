use std::fmt::{Debug, Write};

use ash::vk;


#[cfg(feature = "glsl")]
pub fn glsl(input: proc_macro2::TokenStream) -> proc_macro2::TokenStream {
    use std::fmt::Debug;

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

    let entry_points = reflect_result.into_iter().map(|entry_point| {
        let stage_flags = spirv_stage_to_vk(entry_point.exec_model);
        (entry_point.name, SpirvEntryPoint {
            descriptor_sets: {
                let mut sets: Vec<SpirvDescriptorSet> = Vec::new();

                for var in entry_point.vars.iter() {
                    match var {
                        spirq::Variable::Descriptor { desc_ty, desc_bind, nbind, .. }  => {
                            sets.resize(sets.len().max(desc_bind.set() as usize + 1), Default::default());
                            let set = &mut sets[desc_bind.set() as usize];
                            set.bindings.push(SpirvDescriptorSetBinding {
                                binding: desc_bind.bind(),
                                descriptor_type: DescriptorType(spirv_desc_ty_to_vk(desc_ty)),
                                descriptor_count: *nbind,
                                stage_flags,
                            });
                        },
                        _ => (),
                    }
                }
                sets
            },
            push_constant_ranges: entry_point.vars.iter().filter_map(|var| {
                match var {
                    spirq::Variable::PushConstant { name: Some(name), ty } if !name.starts_with('_') => {
                        println!("{:?}", ty);
                        Some(PushConstantRange {
                            stage_flags: ShaderStageFlags(stage_flags),
                            offset: 0,
                            size: 0
                        })
                    },
                    _ => None,
                }
            }).collect(),
        })
    });

    let entry_points_stream = proc_macro2::TokenStream::from_iter(entry_points.flat_map(|(name, entry_point)| {
        use proc_macro2::{TokenTree, Group, Delimiter, Punct, Spacing, TokenStream, Literal};
        let item = proc_macro2::TokenTree::Group(proc_macro2::Group::new(Delimiter::Parenthesis, {
            let serialized = format!("{:?}", entry_point).parse::<proc_macro2::TokenStream>().unwrap();
            quote::quote!{
                #name.to_string(),#serialized
            }
        }));
        return std::iter::once(item).chain(std::iter::once(proc_macro2::TokenTree::Punct(Punct::new(',', Spacing::Alone))))
    }));


    let bin = U32Slice(binary);
    return quote::quote! {{
        use ::rhyolite::shader::{SpirvEntryPoint, SpirvDescriptorSetBinding, SpirvDescriptorSet};
        use ::rhyolite::ash::vk::{PushConstantRange, ShaderStageFlags, DescriptorType};
        ::rhyolite::shader::SpirvShader {
            data: {
                let slice: &[u32] = #bin.as_slice();
                slice
            },
            entry_points: [#entry_points_stream].into(),
        }
    }}
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


#[derive(Clone, Copy)]
struct ShaderStageFlags(vk::ShaderStageFlags);
impl Debug for ShaderStageFlags {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("ShaderStageFlags::")?;
        self.0.fmt(f)?;
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct DescriptorType(vk::DescriptorType);
impl Debug for DescriptorType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("DescriptorType::")?;
        self.0.fmt(f)?;
        Ok(())
    }
}


#[derive(Clone)]
struct SpirvDescriptorSetBinding {
    pub binding: u32,
    pub descriptor_type: DescriptorType,
    pub descriptor_count: u32,
    pub stage_flags: vk::ShaderStageFlags,
}
impl Debug for SpirvDescriptorSetBinding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpirvDescriptorSetBinding")
        .field("binding", &self.binding)
        .field("descriptor_type", &self.descriptor_type)
        .field("descriptor_count", &self.descriptor_count)
        .field("stage_flags", &ShaderStageFlags(self.stage_flags))
        .field("immutable_samplers", &ToVecFmt::<()>(&()))
        .finish()
    }
}

struct ToVecFmt<'a, I: Debug>(&'a I);
impl<'a, I: Debug> Debug for ToVecFmt<'a, I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("vec!")?;
        self.0.fmt(f)?;
        Ok(())
    }
}


#[derive(Clone, Default)]
struct SpirvDescriptorSet {
    pub bindings: Vec<SpirvDescriptorSetBinding>
}
impl Debug for SpirvDescriptorSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpirvDescriptorSet")
            .field("bindings", &ToVecFmt(&self.bindings))
            .finish()?;
        Ok(())
    }
}

struct SpirvEntryPoint {
    pub descriptor_sets: Vec<SpirvDescriptorSet>,
    pub push_constant_ranges: Vec<PushConstantRange>,
}
impl Debug for SpirvEntryPoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpirvEntryPoint")
            .field("descriptor_sets", &ToVecFmt(&self.descriptor_sets))
            .field("push_constant_ranges", &ToVecFmt(&self.push_constant_ranges))
            .finish()?;
        Ok(())
    }
}


#[derive(Debug)]
struct PushConstantRange {
    pub stage_flags: ShaderStageFlags,
    pub offset: u32,
    pub size: u32,
}


fn spirv_stage_to_vk(stage: spirq::ExecutionModel) -> vk::ShaderStageFlags {
    use spirq::ExecutionModel::*;
    match stage {
        Vertex => vk::ShaderStageFlags::VERTEX,
        TessellationControl => vk::ShaderStageFlags::TESSELLATION_CONTROL,
        TessellationEvaluation => vk::ShaderStageFlags::TESSELLATION_EVALUATION,
        Geometry => vk::ShaderStageFlags::GEOMETRY,
        Fragment => vk::ShaderStageFlags::FRAGMENT,
        GLCompute => vk::ShaderStageFlags::COMPUTE,
        Kernel => vk::ShaderStageFlags::COMPUTE,
        TaskNV => vk::ShaderStageFlags::TASK_EXT,
        MeshNV => vk::ShaderStageFlags::MESH_EXT,
        RayGenerationNV => vk::ShaderStageFlags::RAYGEN_KHR,
        IntersectionNV => vk::ShaderStageFlags::INTERSECTION_KHR,
        AnyHitNV => vk::ShaderStageFlags::ANY_HIT_KHR,
        ClosestHitNV => vk::ShaderStageFlags::CLOSEST_HIT_KHR,
        MissNV => vk::ShaderStageFlags::MISS_KHR,
        CallableNV => vk::ShaderStageFlags::CALLABLE_KHR,
    }
}

fn spirv_desc_ty_to_vk(ty: &spirq::DescriptorType) -> vk::DescriptorType {
    use spirq::DescriptorType::*;
    match ty {
        Sampler() => vk::DescriptorType::SAMPLER,
        CombinedImageSampler() => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
        SampledImage() => vk::DescriptorType::SAMPLED_IMAGE,
        StorageImage(_) => vk::DescriptorType::STORAGE_IMAGE,
        UniformTexelBuffer() => vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
        StorageTexelBuffer(_) => vk::DescriptorType::STORAGE_TEXEL_BUFFER,
        UniformBuffer() => vk::DescriptorType::UNIFORM_BUFFER,
        StorageBuffer(_) => vk::DescriptorType::STORAGE_BUFFER,
        InputAttachment(_) => vk::DescriptorType::INPUT_ATTACHMENT,
        AccelStruct() => vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
    }
}