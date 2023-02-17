use syn::parse::{Parse, ParseStream};

extern crate proc_macro;
mod commands;
mod commands_join;
mod transformer;

mod gpu;

struct ExprGpuAsync {
    pub mv: Option<syn::Token![move]>,
    pub stmts: Vec<syn::Stmt>,
}
impl Parse for ExprGpuAsync {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        Ok(ExprGpuAsync {
            mv: input.parse()?,
            stmts: syn::Block::parse_within(input)?,
        })
    }
}

#[proc_macro]
pub fn commands(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    commands::proc_macro_commands(input.into()).into()
}

#[proc_macro]
pub fn gpu(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    gpu::proc_macro_gpu(input.into()).into()
}

#[proc_macro]
pub fn join(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    commands_join::proc_macro_join(input.into()).into()
}
