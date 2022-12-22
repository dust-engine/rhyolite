use crate::transformer::CommandsTransformer;

struct State {
}
impl Default for State {
    fn default() -> Self {
        Self{}
    }
}

impl CommandsTransformer for State {
    fn import(
        &mut self,
        input_tokens: &proc_macro2::TokenStream,
        is_image: bool,
    ) -> proc_macro2::TokenStream {
        todo!()
    }

    fn async_transform(&mut self, input: &syn::ExprAwait) -> syn::Expr {
        let base = input.base.clone();
        syn::Expr::Verbatim(quote::quote!{{
            let mut fut_pinned = std::pin::pin!(#base);
            fut_pinned.as_mut().init(__current_queue);
            loop {
                use ::std::task::Poll::*;
                match fut_pinned.as_mut().record() {
                    Ready(next_queue) => {
                        __current_queue = next_queue;
                        break;
                    },
                    Yielded => {
                        yield;
                    },
                };
            }
        }})
    }
    //               queue
    // Leaf  nodes   the actual queue
    // Join  nodes   None, or the actual queue if all the same.
    // Block nodes   None
    // for blocks, who yields initially?
    // inner block yields. outer block needs to give inner block the current queue, and the inner block choose to yield or not.

    fn macro_transform(&mut self, mac: &syn::ExprMacro) -> syn::Expr {
        todo!()
    }

    fn return_transform(&mut self, ret: &syn::ExprReturn) -> Option<syn::Expr> {
        todo!()
    }
}

pub fn proc_macro_gpu(input: proc_macro2::TokenStream) -> proc_macro2::TokenStream {
    let input = match syn::parse2::<crate::ExprGpuAsync>(input) {
        Ok(input) => input,
        Err(err) => return err.to_compile_error(),
    };
    let mut state = State::default();

    let mut inner_closure_stmts: Vec<_> = input
        .stmts
        .iter()
        .map(|stmt| state.transform_stmt(stmt))
        .collect();
        quote::quote! {
            async_ash::queue::QueueFutureBlock::new(static |__initial_queue| {
                let mut __current_queue: ::async_ash::queue::QueueRef = __initial_queue;
                //let __fut_global_ctx: &mut ::async_ash::future::GlobalContext = unsafe{&mut *__fut_global_ctx};
                #(#inner_closure_stmts)*
                return __current_queue;
            })
        }
}
